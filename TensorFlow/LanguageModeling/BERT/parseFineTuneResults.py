import os, sys
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("./logs_finetuning/") if isfile(join("./logs_finetuning/", f))]

globalStepToken = "INFO:tensorflow:global_step/sec:"
throughputToken = "INFO:tensorflow:Throughput Average (sentences/sec) = "
mrpcAccuracyToken = "INFO:tensorflow:  eval_accuracy = "
squadAccuracyToken = "exact_match"

resultsFile = open("logs_finetuning/finetuningResultsReport.txt", "w+")

for thefile in onlyfiles:
	throughputFlag = False
	tokenCount = 0
	currSum = 0.0
	with open("./logs_finetuning/"+thefile, "r") as curr:
		currLines = curr.readlines()
	for line in currLines:
		if "mrpc" in thefile:
			if mrpcAccuracyToken in line:
				evalAccuracy = float(line.strip().split("=")[-1].strip())
		else:
			if squadAccuracyToken in line:
				evalAccuracy = float(line.strip().split(",")[0].strip().split(":")[-1].strip())

		if not throughputFlag:
			if throughputToken in line:
				throughputFlag = True
				avgThroughput = float(line.strip().split("=")[-1].strip())
		if globalStepToken in line:
			tokenCount += 1
			currValue = float(line.strip().split(":")[-1].strip())
			currSum += currValue
	resultsFile.write("log file: {}\n".format(thefile))
	if tokenCount == 0:
		resultsFile.write("No global_step/sec instances\n")
	else:
		resultsFile.write("num samples: {}, avg global_step/sec value: {}\n".format(tokenCount,currSum/tokenCount))
	resultsFile.write("average throughput: {} sentences/sec\n".format(avgThroughput))
	resultsFile.write("evaluation accuracy: {}\n\n\n".format(evalAccuracy))

resultsFile.close()
