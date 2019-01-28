import matplotlib.pyplot as plt
from parameters import FLAGS

all_runs = []
run = []
run_index = [i for i in range(0,FLAGS.num_epochs)]
titles = []

f = open(FLAGS.log_path,"r")
first_line = True

for line in f:
	if line.strip() == "---TRAINING STARTED---" and first_line != True:
		all_runs.append(run)
		run = []
	elif len(line.strip().split("parameters:")) > 1:
		titles.append(line)
	else:
		words = line.strip().split("validation accuracy: ")
		if len(words) == 2:
			run.append(float(words[1]))

		first_line = False


all_runs.append(run)
f.close()

print(all_runs)

for run in all_runs:
	plt.title(titles[0], fontdict={'fontsize':7})
	plt.plot(run_index, run)
	plt.ylim(ymin=0, ymax=1)
	plt.show()
