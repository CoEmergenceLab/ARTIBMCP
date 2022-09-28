"""
1. initialize OSC client
2. initialize dslr camera
3. initialize Syphon
4. grab image from camera and save it
5. load ml model
6. resize image to proper size for ml model
7. show image to model and get result (e.g. cluster id and response message)
8. perform contour detection & analysis
9. perfrom luminosity analysis
10. send image to Max via Syphon
11. send response message, contour analysis and luminosity analysis via OSC
"""
