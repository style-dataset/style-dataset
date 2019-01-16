import csv
import sys
import pickle 

#################################


# AVAILABLE FUNCTIONS:

# CSV handling
#  		read_csv_to_array(csv_name, header=True)
# 		write_to_csv(csv_name, contents)
#  		write_array_to_csv_as_row(csv_name, row)

#  text files
# 		read_txt_to_string(filename)
#		write_string_to_txt(filename, string)


# visual outputs
		# dotdotdot(char_ = '.')
		

#################################


#################################
# 
#      Callable functions
# 
#################################


# print a dot to the standard out
def dotdotdot(char_ = '.'):
    sys.stdout.write(char_)
    sys.stdout.flush()


# Reads in each line of a CSV into an array
# The header flag indicates whether it has a header line
# Returns an array
def read_csv_to_array(csv_name, header=True):
	arr = []
	csv_name = append_csv_to_filename_if_needed(csv_name)

	with open(csv_name, "r") as cf:
		rd = csv.reader(cf)
		# consume header
		if (header):
		    next(rd)
		for row in rd:
		    if row:
		        arr.append(row)

	return arr


def read_txt_to_string(filename):
	file = open(filename, "r") 
	text = file.read() 
	file.close()
	return text

def write_string_to_txt(filename, string):
	file = open (filename, "w+")
	file.write(string)
	file.close()


# handles maps and lists
def write_to_csv(csv_name, contents):
	try:
		csv_name = append_csv_to_filename_if_needed(csv_name)

		with open(csv_name, 'a') as csvfile:
			wr = csv.writer(csvfile, delimiter=',')
			
			if (type(contents) is dict):
				for key in contents:
					wr.writerow(contents[key])			
			elif (type(contents) is list):
				for row in contents:
					wr.writerow(row)
		return True
	except Exception as e:
		print(e)
		return False


# return false if the writing fails
def write_array_to_csv_as_row(csv_name, row):
	try:
		csv_name = append_csv_to_filename_if_needed(csv_name)
		with open(csv_name, 'a') as csvfile:
			wr = csv.writer(csvfile, delimiter=',')
			wr.writerow(row)
		return True
	except Exception as e:
		print(e)
		return False


def write_to_pickle(filename, contents):
	with open(filename, 'wb') as fp:
		pickle.dump(contents, fp)

def load_from_pickle(filename):
	with open (filename, 'rb') as fp:
	    result = pickle.load(fp)
	return result


#################################
# 
#      Helper functions
# 
#################################

# helper function
# checks if the last four characters are .csv, 
# and appends that filetype if not
def append_csv_to_filename_if_needed(csv_name):
	if (csv_name[-4:] != ".csv"):
		csv_name = csv_name + ".csv"
	return csv_name




