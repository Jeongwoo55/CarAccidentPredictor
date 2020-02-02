import csv

with open("C:/Users/jchoi/Desktop/CarAccidentPredictor/US_Accidents_Dec19.csv") as csv_file:
    with open("C:/Users/jchoi/Desktop/CarAccidentPredictor/Mod US_Accidents_Dec19.csv", 'w') as csvoutput:
        csv_writer = csv.writer(csvoutput, lineterminator='\n')
        csv_reader = csv.reader(csv_file)
        all = []
        row = next(csv_reader)
        all.append(row)

        # differentWeather = {}
        # topWeather = {}
        #
        # for row in csv_reader:
        #     if row[31] not in differentWeather:
        #         differentWeather[row[31]] = 1
        #     else:
        #         differentWeather[row[31]] = differentWeather[row[31]] + 1
        #
        # for i in differentWeather:
        #     if differentWeather[i] > 1000:
        #         topWeather[i] = differentWeather[i]
        #
        # print(differentWeather)
        # print(len(differentWeather))
        # print(topWeather)
        # print(len(topWeather))

        for row in csv_reader:
            if row[31] in "Light Rain":
                row[50] = 1
                for i in range(51, 58):
                    row[i] = 0
            elif row[31] in "Light Drizzle":
                row[50] = 1
                for i in range(51, 58):
                    row[i] = 0
            elif row[31] in "Rain":
                for i in range(50, 52):
                    row[i] = 1
                for i in range(52, 58):
                    row[i] = 0
            elif row[31] in "Heavy Rain":
                for i in range(50, 53):
                    row[i] = 1
                for i in range(53, 58):
                    row[i] = 0
            elif row[31] in "Light Snow":
                for i in range(50, 53):
                    row[i] = 0
                row[53] = 1
                for i in range(54, 58):
                    row[i] = 0
            elif row[31] in "Snow":
                for i in range(50, 53):
                    row[i] = 0
                row[53] = 1
                row[54] = 1
                for i in range(55, 58):
                    row[i] = 0
            elif row[31] in "Heavy Snow":
                for i in range(50, 53):
                    row[i] = 0
                row[53] = 1
                row[54] = 1
                row[55] = 1
                for i in range(56, 58):
                    row[i] = 0
            elif row[31] in "Fog":
                for i in range(50, 56):
                    row[i] = 0
                row[56] = 1
                row[57] = 0
            elif row[31] in "Haze":
                for i in range(50, 57):
                    row[i] = 0
                row[57] = 1
            else:
                for i in range(50, 58):
                    row[i] = 0

        csv_writer.writerows(csv_reader)


        # csv_writer.writerows(all)

# import pandas as pd
#
# d = pd.read_csv("C:/Users/jchoi/Desktop/CarAccidentPredictor/US_Accidents_Dec19.csv", index_col=0)
#
#
#
# for row in d:
#     if row[31] in "Light Rain":
#         row[50] = 1
#         for i in range(51, 58):
#             row[i] = 0
#     elif row[31] in "Light Drizzle":
#         row[50] = 1
#         for i in range(51, 58):
#             row[i] = 0
#     elif row[31] in "Rain":
#         for i in range(50, 52):
#             row[i] = 1
#         for i in range(52, 58):
#             row[i] = 0
#     elif row[31] in "Heavy Rain":
#         for i in range(50, 53):
#             row[i] = 1
#         for i in range(53, 58):
#             row[i] = 0
#     elif row[31] in "Light Snow":
#         for i in range(50, 53):
#             row[i] = 0
#         row[53] = 1
#         for i in range(54, 58):
#             row[i] = 0
#     elif row[31] in "Snow":
#         for i in range(50, 53):
#             row[i] = 0
#         row[53] = 1
#         row[54] = 1
#         for i in range(55, 58):
#             row[i] = 0
#     elif row[31] in "Heavy Snow":
#         for i in range(50, 53):
#             row[i] = 0
#         row[53] = 1
#         row[54] = 1
#         row[55] = 1
#         for i in range(56, 58):
#             row[i] = 0
#     elif row[31] in "Fog":
#         for i in range(50, 56):
#             row[i] = 0
#         row[56] = 1
#         row[57] = 0
#     elif row[31] in "Haze":
#         for i in range(50, 57):
#             row[i] = 0
#         row[57] = 1
#     else:
#         for i in range(50, 58):
#             row[i] = 0
