import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from PIL import Image
import pytesseract
import argparse
import cv2
import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


for filename in os.listdir("../expirement/test/results-Demo"):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        filename = "abc.jpg".format(os.getpid())
        cv2.imwrite(filename, img)
        text = pytesseract.image_to_string(
            Image.open(filename), config='--psm 13')

        os.remove(filename)
        print(text)
        # print(os.path.join(directory, filename))
        continue
    else:
        continue
# img = cv2.resize(image, (1000, 1000))
