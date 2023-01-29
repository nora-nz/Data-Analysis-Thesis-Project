#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#base parameters
number_of_bodyparts = 9
heatmap_resolution = 8
box_length = 40 #cms
heatmap_bins = int(box_length/heatmap_resolution)
duration = (19 * 60) + 49 #secs
top_left_coords = [11, 23]
bottom_right_coords = [296, 295]

#calculate physical coordinate locations
x_ratio = (bottom_right_coords[0] - top_left_coords[0])/box_length
y_ratio = (bottom_right_coords[1] - top_left_coords[1])/box_length
top_right_coords = [bottom_right_coords[0]/x_ratio, top_left_coords[1]/y_ratio]

#import data into pandas dataframe
path1 = "\\path\\to\\video\\file.avi"
path2 = "\\path\\to\\output\\figures\\folder\\"
csv = pd.read_csv(path1 + "name_of_DLC_output_corresponding_to_avi_in_path1.csv")
csv_working_copy = csv
# folder1 = path2 + "simple-trajectory\\"                     #did not end up using these in the final write-up
# folder2 = path2 + "heatmap\\"                               #did not end up using these in the final write-up
# folder3 = path2 + "cumulative-travel-distance\\"            #did not end up using these in the final write-up

bodyparts = ["nose", "nose", "right ear", "right ear", "left ear", "left ear", "midpoint", "midpoint", "right hind leg", "right hind leg", "left hind leg", "left hind leg", "hip", "hip", "mid tail", "mid tail", "tailtip", "tailtip"]

frame_number = csv["scorer"].tolist()
frame_number.remove("bodyparts")
frame_number.remove("coords")

time_resolution = duration/len(frame_number)

time = np.linspace(time_resolution, duration, len(frame_number), endpoint=True)

columns = csv_working_copy.columns

#delete likelyhood columns
for bodypart in range(0, number_of_bodyparts):
    number = 3*bodypart+3
    del csv_working_copy[columns[number]]

#delete first 3 rows
csv_working_copy.drop([0,1], axis=0, inplace=True)


csv_working_copy = csv_working_copy.astype(float)

#convert values to cm
for column in range(0, len(csv_working_copy.columns)):
    x = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    if column in x:
        csv_working_copy[csv_working_copy.columns[column]] = csv_working_copy[csv_working_copy.columns[column]].div(x_ratio)
    elif column in y:
        csv_working_copy[csv_working_copy.columns[column]] = csv_working_copy[csv_working_copy.columns[column]].div(y_ratio)

# #trajectory - #did not end up using these in the final write-up
# for column in range(1, len(csv_working_copy.columns)):
#     x = [1, 3, 5, 7, 9, 11, 13, 15, 17]
#     y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
#     if column in x:
#         xcoords = box_length - (top_right_coords[0] - csv_working_copy[csv_working_copy.columns[column]])
#         ycoords = box_length - (csv_working_copy[csv_working_copy.columns[column+1]] - top_right_coords[1])
#         plt.plot(xcoords, ycoords)
#         plt.axis([-2, 42, -2, 42])
#         plt.title(bodyparts[column])
#         plt.xlabel("x (cm)")
#         plt.ylabel("y (cm)")
#         plt.savefig(folder1 + bodyparts[column])
#         plt.cla()
#         plt.hist2d(xcoords, ycoords, bins=heatmap_bins)
#         plt.axis("off")
#         plt.savefig(folder2 + bodyparts[column])
#         plt.cla()

# distances, speeds, cumulative distance - #did not end up using these in the final write-up
distances = []
speeds = []
xsquares = []
ysquares = []
cumulative = []

for column in range(0, len(csv_working_copy.columns)):
    x = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    if column in x:
        xdiff = csv_working_copy[csv_working_copy.columns[column]].diff()
        xsquares.append(xdiff**2)
    elif column in y:
        ydiff = csv_working_copy[csv_working_copy.columns[column]].diff()
        ysquares.append(ydiff**2)

for index, item in enumerate(xsquares):
    sums = item.add(ysquares[index])
    sqrt = np.sqrt((sums))
    cumulativesum = np.cumsum(sqrt)/100
    cumulative.append(cumulativesum)
    sqrtsum = np.sum(sqrt)/100
    distances.append(sqrtsum)

for item in distances:
    speed = item/(duration/60)
    speeds.append(speed)

average_distance = sum(distances)/9
average_speed = sum(speeds)/9

# body_parts = ["nose", "right ear", "left ear", "midpoint", "right hind leg", "left hind leg", "hip", "mid tail", "tailtip"] #did not end up using these in the final write-up
# for index, series in enumerate(cumulative):
#     plt.plot(time/60, series)
#     plt.title(body_parts[index])
#     plt.xlabel("time (minutes)")
#     plt.ylabel("cumulative travel distance (cm)")
#     plt.savefig(folder3 + body_parts[index])
#     plt.cla()


edge_width = 5 # in cm -- definition of "edge" and "centre"

num_in_centre = []
for column in np.arange(1, len(csv_working_copy.columns), 2):
    #check every value in column and next column if above are true and if yes append index to list
    in_centre = []
    current_x_coordinates = []
    current_y_coordinates = []
    for index, xitem in csv_working_copy[csv_working_copy.columns[column]].items():
        current_x_coordinates.append(xitem)
    for index, yitem in csv_working_copy[csv_working_copy.columns[column+1]].items():
        current_y_coordinates.append(yitem)
    zipped = list(zip(current_x_coordinates, current_y_coordinates))
    for index, zitem in enumerate(zipped):
        if (zitem[0] >= edge_width) and (zitem[0] <= box_length) and (zitem[1] >= edge_width) and (zitem[1] <= box_length):
            in_centre.append(index)
    num_in_centre.append(len(in_centre))

num_out_centre = []
for item in num_in_centre:
    num_out_centre.append(len(frame_number) - item)

for index, item in enumerate(body_parts):
    print(str((num_in_centre[index] / len(frame_number)) * 100) + "%")
    print(str((num_out_centre[index] / len(frame_number)) * 100) + "%")

# printed averages were manually copied into "Collated Data.csv" because it's just 3 lists

###
### it's all figures from here
###

csv = pd.read_csv("Collated Data.csv")

#genotype & sex vs average distance, average speed, time spent in centre box plots
FAD_pos_F_distances = []
FAD_neg_F_distances = []
FAD_pos_F_speeds = []
FAD_neg_F_speeds = []
FAD_pos_M_distances = []
FAD_neg_M_distances = []
FAD_pos_M_speeds = []
FAD_neg_M_speeds = []
FAD_pos_F_centre = []
FAD_neg_F_centre = []
FAD_pos_M_centre = []
FAD_neg_M_centre = []


for index, item in enumerate(csv[csv.columns[2]]):
    if item == "FAD+" and csv.iloc[index, 4] == "F":
        FAD_pos_F_distances.append(csv.iloc[index, 9])
        FAD_pos_F_speeds.append(csv.iloc[index, 10])
        FAD_pos_F_centre.append(csv.iloc[index, 11])
    elif item == "FAD+" and csv.iloc[index, 4] == "M":
        FAD_pos_M_distances.append(csv.iloc[index, 9])
        FAD_pos_M_speeds.append(csv.iloc[index, 10])
        FAD_pos_M_centre.append(csv.iloc[index, 11])
    elif item == "FAD-" and csv.iloc[index, 4] == "F":
        FAD_neg_F_distances.append(csv.iloc[index, 9])
        FAD_neg_F_speeds.append(csv.iloc[index, 10])
        FAD_neg_F_centre.append(csv.iloc[index, 11])
    else:
        FAD_neg_M_distances.append(csv.iloc[index, 9])
        FAD_neg_M_speeds.append(csv.iloc[index, 10])
        FAD_neg_M_centre.append(csv.iloc[index, 11])

distance_data = [FAD_pos_F_distances, FAD_neg_F_distances, FAD_pos_M_distances, FAD_neg_M_distances]
plt.boxplot(distance_data, showmeans=True, labels=["FAD+ female", "FAD- female", "FAD+ male", "FAD- male"])
plt.title("Average distance travelled")
plt.xlabel("Genotype & Sex")
plt.ylabel("Distance (m)")
plt.ylim(0, 160)
plt.savefig(path + "average distance genotype, sex box")
plt.cla()

speed_data = [FAD_pos_F_speeds, FAD_neg_F_speeds, FAD_pos_M_speeds, FAD_neg_M_speeds]
plt.boxplot(speed_data, showmeans=True, labels=["FAD+ female", "FAD- female", "FAD+ male", "FAD- male"])
plt.title("Average speed")
plt.xlabel("Genotype & Sex")
plt.ylabel("Speed (m/min)")
plt.ylim(0, 8)
plt.savefig(path + "average speed genotype, sex box")
plt.cla()

centre_data = [FAD_pos_F_centre, FAD_neg_F_centre, FAD_pos_M_centre, FAD_neg_M_centre]
plt.boxplot(centre_data, showmeans=True, labels=["FAD+ female", "FAD- female", "FAD+ male", "FAD- male"])
plt.title("Time spent in centre")
plt.xlabel("Genotype & Sex")
plt.ylabel("Time spent in centre (%)")
plt.ylim(0, 80)
plt.savefig(path + "centre genotype, sex box")
plt.cla()

#age&genotype&sex vs average distance, average speed, time spent in the centre scatter plots

age = csv[csv.columns[8]]
distance = csv[csv.columns[9]]
pos_age_F = []
pos_age_M = []
neg_age_F = []
neg_age_M = []
fad_pos_age_dist_F = []
fad_pos_age_dist_M = []
fad_neg_age_dist_F = []
fad_neg_age_dist_M = []
fad_pos_age_speed_F = []
fad_pos_age_speed_M = []
fad_neg_age_speed_F = []
fad_neg_age_speed_M = []
fad_pos_age_centre_F = []
fad_pos_age_centre_M = []
fad_neg_age_centre_F = []
fad_neg_age_centre_M = []

for index, item in enumerate(csv[csv.columns[8]]):
    if csv.iloc[index, 2] == "FAD+" and csv.iloc[index, 4] == "F":
        fad_pos_age_dist_F.append(csv.iloc[index, 9])
        fad_pos_age_speed_F.append(csv.iloc[index, 10])
        fad_pos_age_centre_F.append(csv.iloc[index, 11])
        pos_age_F.append(item)
    elif csv.iloc[index, 2] == "FAD+" and csv.iloc[index, 4] == "M":
        fad_pos_age_dist_M.append(csv.iloc[index, 9])
        fad_pos_age_speed_M.append(csv.iloc[index, 10])
        fad_pos_age_centre_M.append(csv.iloc[index, 11])
        pos_age_M.append(item)
    elif csv.iloc[index, 2] == "FAD-" and csv.iloc[index, 4] == "F":
        fad_neg_age_dist_F.append(csv.iloc[index, 9])
        fad_neg_age_speed_F.append(csv.iloc[index, 10])
        fad_neg_age_centre_F.append(csv.iloc[index, 11])
        neg_age_F.append(item)
    else:
        fad_neg_age_dist_M.append(csv.iloc[index, 9])
        fad_neg_age_speed_M.append(csv.iloc[index, 10])
        fad_neg_age_centre_M.append(csv.iloc[index, 11])
        neg_age_M.append(item)


plt.scatter(pos_age_F, fad_pos_age_dist_F, marker="^", c="pink", label="FAD+ females")
plt.scatter(neg_age_F, fad_neg_age_dist_F, marker="^", c="red", label="FAD- females")
plt.scatter(pos_age_M, fad_pos_age_dist_M, marker="o", c="purple", label="FAD+ males")
plt.scatter(neg_age_M, fad_neg_age_dist_M, marker="o", c="blue", label="FAD- males")

z1 = np.polyfit(pos_age_F, fad_pos_age_dist_F, 1)
p1 = np.poly1d(z1)
z2 = np.polyfit(neg_age_F, fad_neg_age_dist_F, 1)
p2 = np.poly1d(z2)
z3 = np.polyfit(pos_age_M, fad_pos_age_dist_M, 1)
p3 = np.poly1d(z3)
z4 = np.polyfit(neg_age_M, fad_neg_age_dist_M, 1)
p4 = np.poly1d(z4)

plt.plot(pos_age_F, p1(pos_age_F), color="pink")
plt.plot(neg_age_F, p2(neg_age_F), color="red")
plt.plot(pos_age_M, p3(pos_age_M), color="purple")
plt.plot(neg_age_M, p4(neg_age_M), color="blue")

plt.annotate("FAD+ F r^2 = 0.8388\np-value = 0.049", (6.5, 135))
plt.annotate("FAD- F r^2 = 0.9887\np-value = 0.040", (6.5, 50))
plt.annotate("FAD+ M r^2 = 0.3915\np-value = 0.078", (6.5, 100))
plt.annotate("FAD- M r^2 = 0.4980\np-value = 0.070", (6.5, 70))

plt.legend(loc='lower left')
plt.title("Average distances travelled\n by mice of different ages, sexes and genotypes")
plt.xlabel("Age (month)")
plt.ylabel("Average distance (m)")
plt.ylim(0, 160)
plt.xticks([1.3, 3.6, 4.2, 4.8, 5.3, 8.5])
plt.savefig(path + "average distance travelled age, genotype, sex scatter")
plt.cla()

plt.scatter(pos_age_F, fad_pos_age_speed_F, marker="^", c="pink", label="FAD+ females")
plt.scatter(neg_age_F, fad_neg_age_speed_F, marker="^", c="red", label="FAD- females")
plt.scatter(pos_age_M, fad_pos_age_speed_M, marker="o", c="purple", label="FAD+ males")
plt.scatter(neg_age_M, fad_neg_age_speed_M, marker="o", c="blue", label="FAD- males")

z5 = np.polyfit(pos_age_F, fad_pos_age_speed_F, 1)
p5 = np.poly1d(z5)
z6 = np.polyfit(neg_age_F, fad_neg_age_speed_F, 1)
p6 = np.poly1d(z6)
z7 = np.polyfit(pos_age_M, fad_pos_age_speed_M, 1)
p7 = np.poly1d(z7)
z8 = np.polyfit(neg_age_M, fad_neg_age_speed_M, 1)
p8 = np.poly1d(z8)

plt.plot(pos_age_F, p5(pos_age_F), color="pink")
plt.plot(neg_age_F, p6(neg_age_F), color="red")
plt.plot(pos_age_M, p7(pos_age_M), color="purple")
plt.plot(neg_age_M, p8(neg_age_M), color="blue")

plt.annotate("FAD+ F r^2 = 0.8436\np-value = 0.060", (6.5, 7))
plt.annotate("FAD- F r^2 = 0.9947\np-value = 0.048", (6.5, 3))
plt.annotate("FAD+ M r^2 = 0.7400\np-value = 0.096", (6.5, 5))
plt.annotate("FAD- M r^2 = 0.3938\np-value = 0.084", (6.5, 4))

plt.legend(loc='lower left')
plt.title("Average speed of mice of different ages, sexes and genotypes")
plt.xlabel("Age (month)")
plt.ylabel("Average speed (m/min)")
plt.ylim(0, 8)
plt.xticks([1.3, 3.6, 4.2, 4.8, 5.3, 8.5])
plt.savefig(path + "average speed age, genotype, sex scatter")
plt.cla()

plt.scatter(pos_age_F, fad_pos_age_centre_F, marker="^", c="pink", label="FAD+ females")
plt.scatter(neg_age_F, fad_neg_age_centre_F, marker="^", c="red", label="FAD- females")
plt.scatter(pos_age_M, fad_pos_age_centre_M, marker="o", c="purple", label="FAD+ males")
plt.scatter(neg_age_M, fad_neg_age_centre_M, marker="o", c="blue", label="FAD- males")

z9 = np.polyfit(pos_age_F, fad_pos_age_centre_F, 1)
p9 = np.poly1d(z9)
z10 = np.polyfit(neg_age_F, fad_neg_age_centre_F, 1)
p10 = np.poly1d(z10)
z11 = np.polyfit(pos_age_M, fad_pos_age_centre_M, 1)
p11 = np.poly1d(z11)
z12 = np.polyfit(neg_age_M, fad_neg_age_centre_M, 1)
p12 = np.poly1d(z12)

plt.plot(pos_age_F, p9(pos_age_F), color="pink")
plt.plot(neg_age_F, p10(neg_age_F), color="red")
plt.plot(pos_age_M, p11(pos_age_M), color="purple")
plt.plot(neg_age_M, p12(neg_age_M), color="blue")

plt.annotate("FAD+ F r^2 = 0.8182\np-value = 0.205", (6.5, 70))
plt.annotate("FAD- F r^2 = 0.1134\np-value = 0.232", (6.5, 60))
plt.annotate("FAD+ M r^2 = 0.1271\np-value = 0.265", (6.5, 50))
plt.annotate("FAD- M r^2 = 0.0342\np-value = 0.248", (6.5, 40))

plt.legend(loc='lower left')
plt.title("Percentage of time spent in centre\n of mice of different ages, sexes and genotypes")
plt.xlabel("Age (month)")
plt.ylabel("Time spent in centre (%)")
plt.ylim(0, 80)
plt.xticks([1.3, 3.6, 4.2, 4.8, 5.3, 8.5])
plt.savefig(path + "centre age, genotype, sex scatter")
plt.cla()

df = csv.iloc[:, 12:19]
df.insert(0, "AnimalID", csv.iloc[:,0])

#% of time spent in identified behaviours - stacked barchart
colours=plt.cm.viridis(np.linspace(0.3, 1, 7))
celltext=[]
for animal in range(0,19):
    roundnumbers = []
    for item in csv.iloc[animal, 12:19].tolist():
        roundnumbers.append(round(item, 2))
    celltext.append(roundnumbers)


df.plot(x="AnimalID", kind="bar", stacked=True, color=colours, figsize=(8,9))
plt.title(label="Percentage of time spent in identified behaviours", loc="left", pad=20)
plt.ylabel("Percentage of time (%)")
plt.legend(bbox_to_anchor=(1.05, 0.9))
the_table = plt.table(cellText=celltext,
                      rowLabels=csv.iloc[:, 0].tolist(),
                      colColours=colours,
                      colLabels=csv.columns[12:19].tolist(),
                      bbox=[-0.14, -2.1, 1.6, 1.8])
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
plt.subplots_adjust(bottom=0.6, right=0.7)
plt.savefig(path + "stacked bar chart all animals")
plt.show()
plt.cla()

#sex vs behaviours - stacked barchart
isfem = csv["Sex"] == "F"
femdf = csv[isfem]

ismal = csv["Sex"] == "M"
maldf = csv[ismal]


favg = []
for column in femdf[femdf.columns[12:19]]:
    favg.append(femdf[column].mean())
favg.insert(0, "F")

mavg = []
for column in maldf[maldf.columns[12:19]]:
    mavg.append(maldf[column].mean())
mavg.insert(0, "M")


columns1 = femdf.columns[12:19].tolist()
columns1.insert(0, "Sex")

lst = []
for item in favg[1:8]:
    lst.append(round(item, 2))

lst2 = []
for item in mavg[1:8]:
    lst2.append(round(item, 2))

celltext1 = [lst, lst2]

newdf = pd.DataFrame(columns=columns1, index=[1,2])
newdf.loc[1] = favg
newdf.loc[2] = mavg

colours1 = plt.cm.winter(np.linspace(0.3, 1, 7))

newdf.plot(x="Sex", kind="bar", stacked=True, color=colours1)
plt.title(label="Percentage of time spent in identified behaviours by sex", loc="left", pad=20)
plt.ylabel("Percentage of time (%)")
plt.legend(bbox_to_anchor=(1.05, 0.8))
plt.subplots_adjust(bottom=0.3, right=0.65)
the_table = plt.table(cellText=celltext1,
                      rowLabels=["F", "M"],
                      colColours=colours1,
                      colLabels=columns1[1:8],
                      bbox=[-0.18, -0.45, 1.8, 0.25])
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
plt.savefig(path + "stacked bar chart sex")
plt.cla()

#genotype vs behaviours - stacked barchart

ispos = csv["Genotype"] == "FAD+"
posdf = csv[ispos]

isneg = csv["Genotype"] == "FAD-"
negdf = csv[isneg]

posavg = []
for column in posdf[posdf.columns[12:19]]:
    posavg.append(posdf[column].mean())
posavg.insert(0, "FAD+")

negavg = []
for column in negdf[negdf.columns[12:19]]:
    negavg.append(negdf[column].mean())
negavg.insert(0,"FAD-")

lst3 = []
for item in posavg[1:8]:
    lst3.append(round(item, 2))

lst4 = []
for item in negavg[1:8]:
    lst4.append(round(item, 2))

celltext3 = [lst3, lst4]

colours2 = plt.cm.magma(np.linspace(0.3, 1, 7))

columns2 = posdf.columns[12:19].tolist()
columns2.insert(0, "Genotype")


newdf2 = pd.DataFrame(columns=columns2, index=[1,2])
newdf2.loc[1] = posavg
newdf2.loc[2] = negavg

newdf2.plot(x="Genotype", kind="bar", stacked=True, color=colours2)
plt.title(label="Percentage of time spent in identified behaviours by genotype", loc="left", pad=20)
plt.ylabel("Percentage of time (%)")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.subplots_adjust(bottom=0.4, right=0.55)
the_table = plt.table(cellText=celltext3,
                      rowLabels=["FAD+", "FAD-"],
                      colColours=colours2,
                      colLabels=columns2[1:8],
                      bbox=[-0.08, -0.6, 2.13, 0.25])
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
plt.show()
plt.cla()

#sex&genotype vs behaviours - stacked barchart

posFdf = posdf[isfem]
negFdf = negdf[isfem]
posMdf = posdf[ismal]
negMdf = negdf[ismal]


posFavg = []
for column in posFdf[posFdf.columns[12:19]]:
    posFavg.append(posFdf[column].mean())
posFavg.insert(0, "FAD+ Female")

negFavg = []
for column in negFdf[negFdf.columns[12:19]]:
    negFavg.append(negFdf[column].mean())
negFavg.insert(0,"FAD- Female")

###

posMavg = []
for column in posMdf[posMdf.columns[12:19]]:
    posMavg.append(posMdf[column].mean())
posMavg.insert(0, "FAD+ Male")

negMavg = []
for column in negMdf[negMdf.columns[12:19]]:
    negMavg.append(negMdf[column].mean())
negMavg.insert(0,"FAD- Male")

colours3 = plt.cm.cool(np.linspace(0.3, 1, 7))
columns3 = posFdf.columns[12:19].tolist()
columns3.insert(0, "Genotype & Sex")


newdf3 = pd.DataFrame(columns=columns3, index=[1,2,3,4])
newdf3.loc[1] = posFavg
newdf3.loc[2] = negFavg
newdf3.loc[3] = posMavg
newdf3.loc[4] = negMavg

ct1 = []
for item in posFavg[1:8]:
    ct1.append(round(item, 2))

ct2 = []
for item in negFavg[1:8]:
    ct2.append(round(item, 2))

ct3 = []
for item in posMavg[1:8]:
    ct3.append(round(item, 2))

ct4 = []
for item in negMavg[1:8]:
    ct4.append(round(item, 2))

celltext4 = [ct1, ct2, ct3, ct4]
col_w=[]
for item in celltext4:
    for i in item:
        col_w.append(0.3)
for item in range(1,4):
    col_w.append(0.3)

newdf3.plot(x="Genotype & Sex", kind="bar", stacked=True, color=colours3,figsize=(9,5))
plt.title(label="Percentage of time spent in identified behaviours by genotype and sex", loc="left", pad=10)
plt.ylabel("Percentage of time (%)")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tick_params(axis='both', which='major', labelsize=8)
plt.subplots_adjust(right=0.55, bottom=0.4)
the_table = plt.table(cellText=celltext4,
                      rowLabels=["FAD+ Female", "FAD- Female", "FAD+ Male", "FAD- Male"],
                      rowLoc="right",
                      colColours=colours3,
                      colLabels=columns3[1:9],
                      colWidths=col_w,
                      bbox=[0.01, -0.8, 2, 0.3])
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
plt.show()
plt.cla()

#each behaviour by genotype - box plots


for behaviour in range(12,19):
    current_data_pos = []
    current_data_neg = []
    for index, animal in enumerate(csv[csv.columns[0]]):
        if csv.iloc[index, 2] == "FAD+":
            current_data_pos.append(csv.iloc[index, behaviour])
        else:
            current_data_neg.append(csv.iloc[index, behaviour])
    all_current = [current_data_pos, current_data_neg]
    plt.boxplot(all_current, showmeans=True, labels=["FAD+", "FAD-"])
    plt.subplots_adjust(right=0.85, bottom=0.4, left=0.15)
    pos = []
    for item in current_data_pos:
        pos.append(round(item, 2))
    neg = []
    for item in current_data_neg:
        neg.append(round(item, 2))
    neg.append("N/A")
    celltext5 = [pos, neg]
    plt.title("Percentage of time spent " + str(csv.columns[behaviour]) + " by genotype")
    table = plt.table(cellText=celltext5,
                      rowLabels=["FAD+", "FAD-"],
                      bbox=[-0.05, -0.5, 1.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.ylabel("Percentage of time (%)")
    plt.xlabel("Genotype")
    plt.ylim(0, 100)
    plt.cla()


#each behaviour by sex - box plots

for behaviour in range(12,19):
    current_data_F = []
    current_data_M = []
    for index, animal in enumerate(csv[csv.columns[0]]):
        if csv.iloc[index, 4] == "F":
            current_data_F.append(csv.iloc[index, behaviour])
        else:
            current_data_M.append(csv.iloc[index, behaviour])
    all_current_sex = [current_data_F, current_data_M]
    plt.boxplot(all_current_sex, showmeans=True, labels=["F", "M"])
    plt.subplots_adjust(left=0.15, bottom=0.4, right=0.85)
    fem1 = []
    for item in current_data_F:
        fem1.append(round(item, 2))
    mal1 = []
    for item in current_data_M:
        mal1.append(round(item, 2))
    for item in range(1,4):
        fem1.append("N/A")
    col_w1=[]
    for item in fem1:
        col_w1.append(0.1)
    for item in mal1:
        col_w1.append(0.1)
    celltext6 = [fem1, mal1]
    plt.title("Percentage of time spent "+ str(csv.columns[behaviour]) + " by sex")
    table = plt.table(cellText=celltext6,
                      rowLabels=["Female", "Male"],
                      colWidths=col_w1,
                      bbox=[-0.05, -0.5, 1.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.ylim(0, 100)
    plt.ylabel("Percentage of time (%)")
    plt.xlabel("Sex")
    plt.cla()


#each behaviour by genotype&sex - box plots

for behaviour in range(12,19):
    current_data_Fpos = []
    current_data_Fneg = []
    current_data_Mpos = []
    current_data_Mneg = []
    for index, animal in enumerate(csv[csv.columns[0]]):
        if csv.iloc[index, 4] == "F" and csv.iloc[index, 2] == "FAD+":
            current_data_Fpos.append(csv.iloc[index, behaviour])
        elif csv.iloc[index, 4] == "F" and csv.iloc[index, 2] == "FAD-":
            current_data_Fneg.append(csv.iloc[index, behaviour])
        elif csv.iloc[index, 4] == "M" and csv.iloc[index, 2] == "FAD+":
            current_data_Mpos.append(csv.iloc[index, behaviour])
        else:
            current_data_Mneg.append(csv.iloc[index, behaviour])
    all_current_sex_genotype = [current_data_Fpos, current_data_Fneg, current_data_Mpos, current_data_Mneg]
    plt.boxplot(all_current_sex_genotype, showmeans=True, labels=["FAD+ F", "FAD- F", "FAD+ M", "FAD- M"])
    plt.subplots_adjust(left=0.15, bottom=0.4, right=0.85)
    plt.title("Percentage of time spent " + str(csv.columns[behaviour]) + " by genotype and sex")
    one = []
    for item in current_data_Fpos:
        one.append(round(item, 2))
    two = []
    for item in current_data_Fneg:
        two.append(round(item, 2))
    three = []
    for item in current_data_Mpos:
        three.append(round(item, 2))
    four = []
    for item in current_data_Mneg:
        four.append(round(item, 2))
    for item in range(1,3):
        one.append("N/A")
        two.append("N/A")
    four.append("N/A")
    celltext7 = [one, two, three, four]
    table = plt.table(cellText=celltext7,
                      rowLabels=["FAD+ Female", "FAD- Female", "FAD+ Male", "FAD- Male"],
                      bbox=[0.1, -0.8, 1.1, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.ylabel("Percentage of time (%)")
    plt.xlabel("Genotype & Sex")
    plt.ylim(0, 100)
    plt.cla()