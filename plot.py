import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import keras
from keras.models import load_model


for i in range(1,len(sys.argv)):
	model_name = sys.argv[i]
	print(model_name)
	model = load_model(model_name)
	print("")
	print("")
	print(model_name)
	print(model.summary())

	with open(model_name + ".pickle", "rb") as input_file:
		hist = pickle.load(input_file)

	loss = hist["loss"]
	acc = hist["acc"]

	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(range(1,len(loss)+1),loss,label="Loss")
	ax1.legend()
	ax2.plot(range(1,len(acc)+1),acc,label="Accuracy")
	ax2.legend()
	plt.xlabel("Epoch")
	#plt.show()
	plt.savefig(model_name+".png")
