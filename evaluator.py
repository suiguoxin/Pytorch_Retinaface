import os

cmd = 'python test_widerface.py --trained_model "./weights/mobilenet0.25_Final.pth" --network mobile0.25 \
    && cd ./widerface_evaluate \
    && python setup.py build_ext --inplace \
    && python evaluation.py'

os.system(cmd)