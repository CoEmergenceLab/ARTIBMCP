"""
1. initialize OSC client
2. initialize uvc thermal camera
3. initialize Syphon
4. grab image from camera and save it
5. load ml model
6. resize image to propoer size for ml model
7. show image to model and get result (e.g. cluster id and response message)
8. perform contour detection & analysis
9. send image to Max via Syphon
10. send response message and contour analysis via OSC
"""

from uvctypes import *
import cv2
import time
import math
import os
import argparse
import numpy as np
import sys
import pickle
# Importing the model and methods for transfer learning
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pythonosc import udp_client, osc_message_builder, osc_bundle_builder

sys.path.insert(0, "../utils/")  # adding utils folder to the system path
import Syphon
import glfw

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

BUF_SIZE = 2
q = Queue(BUF_SIZE)


def py_frame_callback(frame, userptr):
    array_pointer = cast(
        frame.contents.data,
        POINTER(c_uint16 * (frame.contents.width * frame.contents.height)),
    )
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
        frame.contents.height, frame.contents.width
    )  # no copy

    # data = np.fromiter(
    #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
    # ).reshape(
    #   frame.contents.height, frame.contents.width, 2
    # ) # copy

    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    if not q.full():
        q.put(data)


PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)


def generate_color_map():
    """
    Conversion of the color map from GetThermal to a numpy LUT:
        https://github.com/groupgets/GetThermal/blob/bb467924750a686cc3930f7e3a253818b755a2c0/src/dataformatter.cpp#L6
    """

    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    colormap_ironblack = [
        255, 255, 255, 253, 253, 253, 251, 251, 251, 249, 249, 249, 247, 247,
        247, 245, 245, 245, 243, 243, 243, 241, 241, 241, 239, 239, 239, 237,
        237, 237, 235, 235, 235, 233, 233, 233, 231, 231, 231, 229, 229, 229,
        227, 227, 227, 225, 225, 225, 223, 223, 223, 221, 221, 221, 219, 219,
        219, 217, 217, 217, 215, 215, 215, 213, 213, 213, 211, 211, 211, 209,
        209, 209, 207, 207, 207, 205, 205, 205, 203, 203, 203, 201, 201, 201,
        199, 199, 199, 197, 197, 197, 195, 195, 195, 193, 193, 193, 191, 191,
        191, 189, 189, 189, 187, 187, 187, 185, 185, 185, 183, 183, 183, 181,
        181, 181, 179, 179, 179, 177, 177, 177, 175, 175, 175, 173, 173, 173,
        171, 171, 171, 169, 169, 169, 167, 167, 167, 165, 165, 165, 163, 163,
        163, 161, 161, 161, 159, 159, 159, 157, 157, 157, 155, 155, 155, 153,
        153, 153, 151, 151, 151, 149, 149, 149, 147, 147, 147, 145, 145, 145,
        143, 143, 143, 141, 141, 141, 139, 139, 139, 137, 137, 137, 135, 135,
        135, 133, 133, 133, 131, 131, 131, 129, 129, 129, 126, 126, 126, 124,
        124, 124, 122, 122, 122, 120, 120, 120, 118, 118, 118, 116, 116, 116,
        114, 114, 114, 112, 112, 112, 110, 110, 110, 108, 108, 108, 106, 106,
        106, 104, 104, 104, 102, 102, 102, 100, 100, 100, 98, 98, 98, 96, 96,
        96, 94, 94, 94, 92, 92, 92, 90, 90, 90, 88, 88, 88, 86, 86, 86, 84, 84,
        84, 82, 82, 82, 80, 80, 80, 78, 78, 78, 76, 76, 76, 74, 74, 74, 72, 72,
        72, 70, 70, 70, 68, 68, 68, 66, 66, 66, 64, 64, 64, 62, 62, 62, 60, 60,
        60, 58, 58, 58, 56, 56, 56, 54, 54, 54, 52, 52, 52, 50, 50, 50, 48, 48,
        48, 46, 46, 46, 44, 44, 44, 42, 42, 42, 40, 40, 40, 38, 38, 38, 36, 36,
        36, 34, 34, 34, 32, 32, 32, 30, 30, 30, 28, 28, 28, 26, 26, 26, 24, 24,
        24, 22, 22, 22, 20, 20, 20, 18, 18, 18, 16, 16, 16, 14, 14, 14, 12, 12,
        12, 10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 9,
        2, 0, 16, 4, 0, 24, 6, 0, 31, 8, 0, 38, 10, 0, 45, 12, 0, 53, 14, 0,
        60, 17, 0, 67, 19, 0, 74, 21, 0, 82, 23, 0, 89, 25, 0, 96, 27, 0, 103,
        29, 0, 111, 31, 0, 118, 36, 0, 120, 41, 0, 121, 46, 0, 122, 51, 0, 123,
        56, 0, 124, 61, 0, 125, 66, 0, 126, 71, 0, 127, 76, 1, 128, 81, 1, 129,
        86, 1, 130, 91, 1, 131, 96, 1, 132, 101, 1, 133, 106, 1, 134, 111, 1,
        135, 116, 1, 136, 121, 1, 136, 125, 2, 137, 130, 2, 137, 135, 3, 137,
        139, 3, 138, 144, 3, 138, 149, 4, 138, 153, 4, 139, 158, 5, 139, 163,
        5, 139, 167, 5, 140, 172, 6, 140, 177, 6, 140, 181, 7, 141, 186, 7,
        141, 189, 10, 137, 191, 13, 132, 194, 16, 127, 196, 19, 121, 198, 22,
        116, 200, 25, 111, 203, 28, 106, 205, 31, 101, 207, 34, 95, 209, 37,
        90, 212, 40, 85, 214, 43, 80, 216, 46, 75, 218, 49, 69, 221, 52, 64,
        223, 55, 59, 224, 57, 49, 225, 60, 47, 226, 64, 44, 227, 67, 42, 228,
        71, 39, 229, 74, 37, 230, 78, 34, 231, 81, 32, 231, 85, 29, 232, 88,
        27, 233, 92, 24, 234, 95, 22, 235, 99, 19, 236, 102, 17, 237, 106, 14,
        238, 109, 12, 239, 112, 12, 240, 116, 12, 240, 119, 12, 241, 123, 12,
        241, 127, 12, 242, 130, 12, 242, 134, 12, 243, 138, 12, 243, 141, 13,
        244, 145, 13, 244, 149, 13, 245, 152, 13, 245, 156, 13, 246, 160, 13,
        246, 163, 13, 247, 167, 13, 247, 171, 13, 248, 175, 14, 248, 178, 15,
        249, 182, 16, 249, 185, 18, 250, 189, 19, 250, 192, 20, 251, 196, 21,
        251, 199, 22, 252, 203, 23, 252, 206, 24, 253, 210, 25, 253, 213, 27,
        254, 217, 28, 254, 220, 29, 255, 224, 30, 255, 227, 39, 255, 229, 53,
        255, 231, 67, 255, 233, 81, 255, 234, 95, 255, 236, 109, 255, 238, 123,
        255, 240, 137, 255, 242, 151, 255, 244, 165, 255, 246, 179, 255, 248,
        193, 255, 249, 207, 255, 251, 221, 255, 253, 235, 255, 255, 24]

    def chunk(ulist, step):
        return map(lambda i: ulist[i : i + step], range(0, len(ulist), step))

    chunks = chunk(colormap_ironblack, 3)

    red = []
    green = []
    blue = []

    for chunk in chunks:
        red.append(chunk[0])
        green.append(chunk[1])
        blue.append(chunk[2])

    lut[:, 0, 0] = blue

    lut[:, 0, 1] = green

    lut[:, 0, 2] = red

    return lut


def ktof(val):
    return 1.8 * ktoc(val) + 32.0


def ktoc(val):
    return (val - 27315) / 100.0


def ftok(val):
    return (((val - 32) * 0.5555555556) * 100) + 27315


def ctok(val):
    return (val * 100.0) + 27315


def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


def display_temperature_f(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(
        img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
    )
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


def display_temperature_c(img, val_k, loc, color):
    val = ktoc(val_k)
    cv2.putText(
        img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
    )
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


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
            "address": "/thermal/response/sound/buffers",
            "arguments": [[2, "i"], [13, "i"]],
        },
        "pitch": {
            "address": "/thermal/response/sound/pitch",
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

def extract_features(target_file):
    """
    This function preprocesses the input target image
    and loades the VGG 16 model to extract features.
    return: Extracted features with dimenions 4096
    """
    # DL model for transfer learning
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    # load the image as a 224x224 array
    img = load_img(target_file, target_size=(224,224))
    # convert from image to numpy array
    img = img_to_array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def preprocess_input_img_for_kmeans(input_target_img_path):
    """
    This function uses the extract features function to preprocess
    and extract the features from the target input image and reduces
    the high dimensional features (4096) to 50.
    return: Reduced features with dimensions 50
    """
    # Extracting the features from the target input image
    img_feat = extract_features(input_target_img_path)
    # Loading PCA pkl file
    pca_pkl = 'kmeans_thermal.pkl'
    pca = pickle.load(open(pca_pkl, 'rb'))
    # Reducing high dimensionality
    reduced_feat = pca.transform(img_feat)
    return reduced_feat




def main():
    prevTime = 0  # will store last time capture time updated
    INTERVAL = args.interval  # default = 15 minutes = 900 seconds
    DIMENSIONS = (320, 240)

    # OSC addresses
    OSC_ADDRESSES = (
        "/thermal/leaf/contour",
        "/thermal/response/sound/buffers",
        "/thermal/response/sound/pitch",
        "/thermal/response/sound/xpos",
        "/thermal/response/sound/ypos",
        "/thermal/response/sound/chopper",
        "/thermal/response/control/water",
        "/thermal/response/control/peg",
        "/thermal/response/control/aba",
        "/thermal/response/cluster",
    )

    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    # ==== Syphon setup details ====
    # Syphon.Server("window and syphon server name", frame size, show)
    syphon_thermal_server = Syphon.Server("ServerThermal", DIMENSIONS, show=False)
    syphon_thermalcv_server = Syphon.Server("ServerThermalCV", DIMENSIONS, show=False)

    # ==== set up libuvc for thermal camera ====
    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)

        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                print("uvc_open error")
                exit(1)

            print("device opened!")

            print_device_info(devh)
            print_device_formats(devh)

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16, frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval))

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            output_dir = "captures"
            color_map = generate_color_map()  # color LUT

            #
            # Min/max value to pin across the LUT
            #
            min_c = ctok(args.range[0])
            max_c = ctok(args.range[1])

            ## ========== MAIN LOOP ========== ##
            try:
                while True:
                    # get a frame from the queue
                    data = q.get(True, 500)
                    if data is None:
                        break

                    data = cv2.resize(data[:, :], DIMENSIONS)  # resize
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)

                    # Dirty-hack to ensure that the LUT is always scaled
                    # against the colors we care about
                    data[0][0] = min_c
                    data[-1][-1] = max_c
                    img = raw_to_8bit(data)
                    if(args.color):
                        img = cv2.LUT(img, color_map)

                    timestr = time.strftime("%Y%m%d-%H%M%S")

                    # Max/min values in the top-left
                    temp_str = "range(" + "{:.2f}, {:.2f}".format(ktoc(minVal), ktoc(maxVal)) + ")"

                    if(args.showtemps):
                        font = cv2.FONT_HERSHEY_PLAIN
                        cv2.putText(img, temp_str, (10, 230), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

                    # time-lapse
                    if(args.timelapse):
                        currTime = time.time()
                        if(currTime-prevTime >= INTERVAL):
                            prevTime = currTime
                            print("capturing image...")
                            cv2.imwrite(os.path.join(output_dir, "{:s}.jpg".format(timestr)), img)
                        # time.sleep(900)  # 900 = 15 mins

                    # display_temperature_c(img, min_c, minLoc, (255, 0, 0))
                    # display_temperature_c(img, max_c, maxLoc, (0, 0, 255))

                    # ==== Load ML model and perform inference ==== #
                    model_pkl_file = 'kmeans_thermal.pkl'
                    # Loading model
                    kmeans_model = pickle.load(
                        open(model_pkl_file, 'rb')
                    )
                    # Preprocessing target input image
                    preprocessed_feat = preprocess_input_img_for_kmeans(
                        img
                    )
                    # Generate prediction cluster ID
                    cluster_id_prediction = kmeans_model.predict(
                        preprocessed_feat
                    )

                    # (make sure image is resized/cropped correctly for the model, e.g. 224x224 for VGG16)

                    ml_bundle_dict = {img: cluster_id_prediction[0]}  # for OSC bundle for ml response

                    # ==== Perform contour detection & analysis ==== #
                    # blur & threshold
                    imgBlur = cv2.medianBlur(img, 5)
                    ret, thresh = cv2.threshold(imgBlur, int(args.threshold), 255, cv2.THRESH_BINARY)

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

                    cv2.imshow('Lepton Radiometry', img)  # show thermal image
                    # draw frame using opengl and send it to Syphon so Max can grab it
                    img_color = cv2.LUT(img, color_map)  # send color image to Syphon/Max
                    img_color_fix = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
                    syphon_thermal_server.draw_and_send(img_color_fix)
                    out2 = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                    syphon_thermalcv_server.draw_and_send(out2)

                    key = cv2.waitKey(50) & 0xFF
                    if key == 27:
                        break

                glfw.terminate()
                cv2.destroyAllWindows()
            finally:
                libuvc.uvc_stop_streaming(devh)

            print("done")
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--range", nargs="*", default=[20, 40], type=float, help="The temperature range to look for")
    parser.add_argument("-c", "--color", action="store_true", help="use grayscale or a color LUT")
    parser.add_argument("-t", "--timelapse", action="store_true", help="timelapse images option")
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        required=False,
        default=900,
        help="timelapse interval (default=900 seconds (15 minutes))",
    )
    parser.add_argument("-s", "--showtemps", action="store_true", help="show temperature ranges")
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
    args = parser.parse_args()
    # OSC
    oscClient = udp_client.UDPClient(args.ip, args.port)
    print(" ")
    sys.exit(main())
