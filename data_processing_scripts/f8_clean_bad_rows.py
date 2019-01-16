# take in a style comparisons data csv from F8 
# strip bad rows based on a set of heuristics
#	if a user contributes a bad row, throw out all their rows
# 	(cannot trust a user's other responses if provides bad faith answers)
#
# save out a csv of good users with their country affiliation
# save out a csv of cleaned rows
##################################################################

import file_utilities as lib
from f8_column_definitions import _worker_id, explanation_b, explanation_c, _channel, text_1, text_2, text_3, _country


#############################
# File to process
path = "" 			#e.g. "style_dataset/raw_data/"
filename = "" 		#e.g. "raw_1.csv"
outpath = path 		# modfiy this if you want it to save clean files somewhere else
#############################

def bad_user(user):
	if user not in bad_users:
		bad_users.append(user)

def is_good_faith(row):
	if "words similar" in row[explanation_b] or "words similar" in row[explanation_c]:
		return False
	if row[explanation_b] is "b" or row[explanation_b] is "B":
		return False
	if row[explanation_c] is "c" or row[explanation_c] is "C":
		return False
	if (row[explanation_b] == "" and row[explanation_c] == "") or row[explanation_b] is "." or row[explanation_c] is ".":
		return False
	if len(row[explanation_b]) > 20 and ( row[explanation_b][:20] in row[text_1] or row[explanation_b][:20] in row[text_2] or row[explanation_b] in row[text_3] ):
		return False
	if len(row[explanation_c]) > 20 and ( row[explanation_c][:20] in row[text_1] or row[explanation_c][:20] in row[text_2] or row[explanation_c] in row[text_3] ):
		return False
	# consistently low quality channels prior to be being banned
	if "content_runner" in row[_channel] or "imerit_india" in row[_channel]:
		return False
	if  row[explanation_b] is "-" or row[explanation_c] is "-":
		return False
	if row[explanation_b] == "b4" or row[explanation_c] == "c4":
		return False
	if row[explanation_b] == "nada" or row[explanation_c] == "nada":
		return False

	return True

good_users = []

# seeded with manually added bad users who respond with nonsense
bad_users = ["360", "10", "508"]

source = path + filename
rows = lib.read_csv_to_array(source)

# go through each row; identify if bad row
# if bad row, record bad user
for row in rows:
	user = row[_worker_id]
	if not is_good_faith(row):
		bad_user(user)

# go through each row; if not by a bad user, 
# add to the clean row list and good user list
clean_rows = []
bad_rows = []
for row in rows:
	if row[_worker_id] not in bad_users:
		clean_rows.append(row)
		if [row[_worker_id], row[_country]] not in good_users:
			good_users.append([row[_worker_id], row[_country]])
	else:
		bad_rows.append(row)

# save out info
lib.write_to_csv(outpath + "cleaned_rows_" + filename, clean_rows)
lib.write_to_csv(outpath + "good_users_with_countries_" + filename, good_users)

# in case you want to see bad rows too:
# lib.write_to_csv(outpath + "bad_rows" + filename, bad_rows)
