"""
1. initialize OSC client
2. initialize dslr camera
3. initialize Syphon
4. grab image from camera and save it
5. load ml model
6. resize image to proper size for ml model
7. show image to model and get result (e.g. cluster id and response message)
8. perform contour detection & analysis
9. perform luminosity analysis
10. send image to Max via Syphon
11. send response message, contour analysis and luminosity analysis via OSC
"""

# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
import math
import logging
import cv2
import numpy as np
import gphoto2 as gp
from pythonosc import udp_client, osc_message_builder, osc_bundle_builder

sys.path.insert(0, "../utils/")  # adding utils folder to the system path
import Syphon
import glfw


def loadImg(s, read_as_float32=False, gray=False):
    if read_as_float32:
        img = cv2.imread(s).astype(np.float32) / 255
    else:
        img = cv2.imread(s)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sendContour(id, perimeter, area, circularity, solidity, addr):
    # send OSC Message representing analysis of a leaf or seedling contour, in the form of:
    # ["/luciferase/plate", id, perimeter, area, circularity, solidity]
    msg = osc_message_builder.OscMessageBuilder(address=addr)
    msg.add_arg(id, arg_type="i")
    msg.add_arg(perimeter, arg_type="f")
    msg.add_arg(area, arg_type="f")
    msg.add_arg(circularity, arg_type="f")
    msg.add_arg(solidity, arg_type="f")
    oscClient.send(msg.build())


def sendContours(bundle_list):
    # send OSC Bundle with OSC messages representing all contour bounding boxes:
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    for b in bundle_list:
        addr, id, perimeter, area, circularity, solidity = b
        msg = osc_message_builder.OscMessageBuilder(address=addr)
        msg.add_arg(id, arg_type="i")
        msg.add_arg(perimeter, arg_type="f")
        msg.add_arg(area, arg_type="f")
        msg.add_arg(circularity, arg_type="f")
        msg.add_arg(solidity, arg_type="f")
        bundle.add_content(msg.build())

    oscClient.send(bundle.build())


def sendResponse(args, addr):
    msg = osc_message_builder.OscMessageBuilder(address=addr)
    # arrays are formatted thusly: [[2, "i"], [13, "i"]]
    # ARG_TYPES: https://github.com/attwad/python-osc/blob/master/pythonosc/osc_message_builder.py
    for arg in args:
        msg.add_arg(arg[0], arg_type=arg[1])

    oscClient.send(msg.build())


def sendResponses(bundle_dict):
    # send OSC Bundle with OSC messages representing all responses from the ml:
    bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    # data is in a dictionary formatted thusly:
    """
    {
        "buffers": {
            "address": "/luciferase/response/sound/buffers",
            "arguments": [[2, "i"], [13, "i"]],
        },
        "pitch": {
            "address": "/luciferase/response/sound/pitch",
            "arguments": [[0.92, "f"]],
        },
        etc...
    }
    """

    # loop through the keys, construct OSC messages and add them to the OSC bundle
    for k in bundle_dict.keys():
        msg = osc_message_builder.OscMessageBuilder(address=k["address"])
        kargs = k["arguments"]
        for arg in kargs:
            msg.add_arg(arg[0], arg_type=arg[1])

        # add this message to the bundle
        bundle.add_content(msg.build())

    # send the bundle
    oscClient.send(bundle.build())


def sendLuminosity(lum, addr):
    # send OSC Message representing the amount of green in the image
    msg = osc_message_builder.OscMessageBuilder(address=addr)
    msg.add_arg(lum, arg_type="f")
    oscClient.send(msg.build())


def setupCamera(target="Memory card"):
    # open camera connection
    camera = gp.Camera()
    camera.init()

    # get configuration tree
    config = camera.get_config()

    # find the capture target config item
    capture_target = config.get_child_by_name("capturetarget")

    # set value to Memory card (default) or Internal RAM
    # value = capture_target.get_value()
    capture_target.set_value(target)
    # set config
    camera.set_config(config)

    return camera


def getGreenPercentage(img):
    # Convert the image to HSV
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # rough RGB ranges: 10, 11, 0 - 202, 175, 29
    # The HSV mask values, defined for the green color ranges
    lowerValues = np.array([38, 148, 16])
    upperValues = np.array([160, 255, 255])

    # Create the HSV mask
    # cv2.inRange is used to binarize the image
    # All the pixels that fall between lowerValues, upperValues will be white
    # All the pixels that do not fall inside this range will be black
    hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)

    # AND mask & input image:
    # All the pixels that are white in the mask will survive the AND operation,
    # all the black pixels will remain black
    # hsvANDMask = cv2.bitwise_and(img, img, mask=hsvMask)

    # You can use the mask to count the number of white pixels.
    # Remember that the white pixels in the mask are those that
    # fall in the defined range, that is, every white pixel corresponds
    # to a green pixel. Divide by the image size and you got the
    # percentage of green pixels in the original image
    # note: img.size gives the number of pixels fro all 3 color channels combined
    ratio_green = cv2.countNonZero(hsvMask) / (img.size / 3)

    # This is the color percent calculation
    colorPercent = ratio_green * 100

    # Print the color percent, use 2 figures past the decimal point
    finalPercent = np.round(colorPercent, 2)
    print("green pixel percentage:", finalPercent)

    return finalPercent


def main():
    prevTime = 0
    INTERVAL = args.interval  # default = 15 minutes = 900 seconds
    DIMENSIONS_MAIN = (720, 480)
    DIMENSIONS_CV = (383, 255)

    # OSC addresses
    OSC_ADDRESSES = (
        "/luciferase/plate/contour",
        "/luciferase/plate/luminosity",
        "/luciferase/response/sound/buffers",
        "/luciferase/response/sound/pitch",
        "/luciferase/response/sound/xpos",
        "/luciferase/response/sound/ypos",
        "/luciferase/response/sound/chopper",
        "/luciferase/response/control/water",
        "/luciferase/response/control/peg",
        "/luciferase/response/control/aba",
        "/luciferase/response/cluster",
    )

    # use Python logging
    logging.basicConfig(
        format="%(levelname)s: %(name)s: %(message)s", level=logging.WARNING
    )
    callback_obj = gp.check_result(gp.use_python_logging())

    # establish connection with digital camera
    camera = setupCamera()

    # ==== Syphon setup details ==== #
    # Syphon.Server("window and syphon server name", frame size, show)
    syphon_luciferase_server = Syphon.Server(
        "ServerLuciferase", DIMENSIONS_MAIN, show=False
    )
    syphon_luciferasecv_server = Syphon.Server(
        "ServerLuciferaseCV", DIMENSIONS_CV, show=False
    )

    ## ========== MAIN LOOP ========== ##
    try:
        while True:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            currTime = time.time()
            if currTime - prevTime >= INTERVAL:
                # ==== grab image from camera and save to disk === #
                prevTime = currTime  # reset time-lapse
                print("Capturing image")
                file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
                print(
                    "Camera file path: {0}/{1}".format(file_path.folder, file_path.name)
                )
                # rename the file with a timestamp
                if file_path.name.lower().endswith(".jpg"):
                    new_filename = "{}.jpg".format(timestr)
                target = os.path.join("./captures", new_filename)
                print("Copying image to", target)
                camera_file = camera.file_get(
                    file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
                )
                camera_file.save(target)
                # load image
                img = loadImg(target)
                # resize image for Syphon
                imgLuciferase = cv2.resize(
                    img, DIMENSIONS_MAIN, interpolation=cv2.INTER_AREA
                )
                # opencv uses bgr so we have to convert
                imgLuciferaseCvt = cv2.cvtColor(imgLuciferase, cv2.COLOR_BGR2RGB)

                # ==== Load ML model and perform inference ==== #
                # (make sure image is resized/cropped correctly for the model, e.g. 224x224 for VGG16)

                ml_bundle_dict = {}  # for OSC bundle for ml response

                # ==== Perform contour detection & analysis ==== #
                # resize image for Syphon
                imgLuciferaseCV = cv2.resize(
                    img, DIMENSIONS_CV, interpolation=cv2.INTER_AREA
                )
                # blur & threshold
                imgBlur = cv2.medianBlur(imgLuciferaseCV, 5)
                ret, thresh = cv2.threshold(
                    imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY
                )

                # find Contours
                contours, hierarchy = cv2.findContours(
                    thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )

                out = np.zeros_like(thresh)

                contours_bundle = []  # for OSC bundle for contour data

                # draw the contours
                for i in range(len(contours)):
                    # -1 in 4th column means it's an external contour
                    if hierarchy[0][i][3] == -1:
                        perimeter = cv2.arcLength(contours[i], True)
                        area = cv2.contourArea(contours[i])
                        circularity = 4 * math.pi * (area / (perimeter * perimeter))
                        hull = cv2.convexHull(contours[i], False)
                        hullArea = cv2.contourArea(hull)
                        solidity = area / hullArea
                        if args.bundle:
                            contours_bundle.append(
                                [
                                    OSC_ADDRESSES[0],
                                    i,
                                    perimeter,
                                    area,
                                    circularity,
                                    solidity,
                                ]
                            )
                        else:
                            sendContour(
                                i,
                                perimeter,
                                area,
                                circularity,
                                solidity,
                                OSC_ADDRESSES[0],
                            )
                        x, y, w, h = cv2.boundingRect(contours[i])
                        cv2.drawContours(out, contours, i, (204, 204, 204), 3)
                        cv2.putText(
                            out,
                            str(i),
                            (x, y - 1),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 0),
                            1,
                        )
                        print("contour " + str(i) + ":")
                        print("  permimeter:" + str(perimeter))
                        print("  area:" + str(area))
                        print("  circularity:" + str(circularity))
                        print("  solidity:" + str(solidity))
                        print(" ")
                        print("---------------------------------")
                        print(" ")

                if args.bundle:
                    sendContours(contours_bundle)

                # ==== Perform color analysis, look amount of green in image ==== #
                # send data via OSC
                sendLuminosity(getGreenPercentage(img), OSC_ADDRESSES[1])

                cv2.imshow("Camera image", img)  # show image
                # draw frame using opengl and send it to Syphon so Max can grab it
                syphon_luciferase_server.draw_and_send(imgLuciferaseCvt)
                out2 = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                syphon_luciferasecv_server.draw_and_send(out2)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        glfw.terminate()
        cv2.destroyAllWindows()
    finally:
        # clean up
        camera.exit()
        print("done")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--threshold",
        default="127",
        help="The cutoff for the threshold algorithm (0-255)",
    )
    parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument(
        "--port", type=int, default=5005, help="The port the OSC server is listening on"
    )
    parser.add_argument(
        "-b",
        "--bundle",
        action="store_true",
        help="Send contours as an OSC Bundle (instead of individual OSC Messages)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        required=False,
        default=900,
        help="timelapse interval (default=900 seconds (15 minutes))",
    )
    args = parser.parse_args()
    # OSC
    oscClient = udp_client.UDPClient(args.ip, args.port)
    print(" ")
    sys.exit(main())
