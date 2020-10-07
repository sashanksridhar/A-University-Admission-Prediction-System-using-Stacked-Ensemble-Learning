from sklearn import preprocessing
import pandas as pd
import csv

with open("E:\\collegerecsys\\data.csv", 'r', encoding='utf8') as r:
    X = []  # train features
    Y = []  # train output probability
    header = []  # training labels
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            header.extend(row)  # get labels alon
            c += 1
            continue
        X.append(row[:len(row) - 1])  # get all features except probability
        Y.append(row[len(row) - 1])  # probability


    header.remove('admit')  # remove output label as input features wont have this

    df = pd.DataFrame(X, columns=header) # training data frame

    df = df.drop(columns=['userName', 'userProfileLink'])  # remove columns that are not needed


    header.remove('userName') # remove column name from label list
    header.remove('userProfileLink')
    # df.dropna(inplace=True)

    major = preprocessing.LabelEncoder() # create encoder
    df['major'] = major.fit_transform(df['major'])   # encode train values
    max_maj = df['major'].max() # calc max between train and test set for that column
    if max_maj!=0:
        df['major'] = df['major'].div(float(max_maj)) # normalize train set


    df['researchExp'] = df['researchExp'].apply(pd.to_numeric) # convert string value read from csv to numeric value
    max_exp =df['researchExp'].max()   # calc max between train and test set for that column
    if max_exp!=0:
        df['researchExp'] = df['researchExp'].div(float(max_exp)) # normalize train set

    df['industryExp'] = df['industryExp'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_inexp = df['industryExp'].max()  # calc max between train and test set for that column
    if max_inexp != 0:
        df['industryExp'] = df['industryExp'].div(float(max_inexp))  # normalize train set

    specialization = preprocessing.LabelEncoder()  # create encoder
    df['specialization'] =specialization.fit_transform(df['specialization'])  # encode train values
    max_spc = df['specialization'].max()  # calc max between train and test set for that column
    if max_spc != 0:
        df['specialization'] = df['specialization'].div(float(max_spc))  # normalize train set

    df['toeflScore'] = df['toeflScore'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_toefl = df['toeflScore'].max()  # calc max between train and test set for that column
    if max_toefl != 0:
        df['toeflScore'] = df['toeflScore'].div(float(max_toefl))  # normalize train set

    program = preprocessing.LabelEncoder()  # create encoder
    df['program'] = program.fit_transform(df['program'])  # encode train values
    max_prgm = df['program'].max()  # calc max between train and test set for that column
    if max_prgm != 0:
        df['program'] = df['program'].div(float(max_prgm))  # normalize train set

    department = preprocessing.LabelEncoder()  # create encoder
    df['department'] =department.fit_transform(df['department'])  # encode train values
    max_dept = df['department'].max()  # calc max between train and test set for that column
    if max_dept!= 0:
        df['department'] = df['department'].div(float(max_dept))  # normalize train set

    df['internExp'] = df['internExp'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_intexp = df['internExp'].max()  # calc max between train and test set for that column
    if max_intexp != 0:
        df['internExp'] = df['internExp'].div(float(max_intexp))  # normalize train set

    df['greV'] = df['greV'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_greV = df['greV'].max()  # calc max between train and test set for that column
    if max_greV != 0:
        df['greV'] = df['greV'].div(float(max_greV))  # normalize train set

    df['greQ'] = df['greQ'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_greQ = df['greQ'].max()  # calc max between train and test set for that column
    if max_greQ != 0:
        df['greQ'] = df['greQ'].div(float(max_greQ))  # normalize train set

    df['journalPubs'] = df['journalPubs'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_pubs = df['journalPubs'].max()  # calc max between train and test set for that column
    if max_pubs != 0:
        df['journalPubs'] = df['journalPubs'].div(float(max_pubs))  # normalize train set

    df['greA'] = df['greA'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_greA = df['greA'].max()  # calc max between train and test set for that column
    if max_greA != 0:
        df['greA'] = df['greA'].div(float(max_greA))  # normalize train set

    df['topperCgpa'] = df['topperCgpa'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_topcgpa = df['topperCgpa'].max()  # calc max between train and test set for that column
    if max_topcgpa != 0:
        df['topperCgpa'] = df['topperCgpa'].div(float(max_topcgpa))  # normalize train set

    term = preprocessing.LabelEncoder()  # create encoder
    df['termAndYear'] = term.fit_transform(df['termAndYear'])  # encode train values
    max_term = df['termAndYear'].max()  # calc max between train and test set for that column
    if max_term != 0:
        df['termAndYear'] = df['termAndYear'].div(float(max_term))  # normalize train set

    df['confPubs'] = df['confPubs'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_conf = df['confPubs'].max()  # calc max between train and test set for that column
    if max_conf != 0:
        df['confPubs'] = df['confPubs'].div(float(max_conf))  # normalize train set

    ug = preprocessing.LabelEncoder()  # create encoder
    df['ugCollege'] = ug.fit_transform(df['ugCollege'])  # encode train values
    max_ug = df['ugCollege'].max()  # calc max between train and test set for that column
    if max_ug != 0:
        df['ugCollege'] = df['ugCollege'].div(float(max_ug))  # normalize train set

    df['cgpa'] = df['cgpa'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_cgpa = df['cgpa'].max()  # calc max between train and test set for that column
    if max_cgpa != 0:
        df['cgpa'] = df['cgpa'].div(float(max_cgpa))  # normalize train set

    df['cgpaScale'] = df['cgpaScale'].apply(pd.to_numeric)  # convert string value read from csv to numeric value
    max_cgpasc = df['cgpaScale'].max()  # calc max between train and test set for that column
    if max_cgpasc != 0:
        df['cgpaScale'] = df['cgpaScale'].div(float(max_cgpasc))  # normalize train set

    univ = preprocessing.LabelEncoder()  # create encoder
    df['univName'] = univ.fit_transform(df['univName'])  # encode train values
    max_univ = df['univName'].max()  # calc max between train and test set for that column
    if max_univ != 0:
        df['univName'] = df['univName'].div(float(max_univ))  # normalize train set

    df['admit'] = Y

    df.dropna(inplace=True)

    l = df.values.tolist() # train dataframe to list


    with open("E:\\collegerecsys\\normal_train.csv", 'w') as w:
        writer = csv.writer(w, lineterminator='\n')
        writer.writerow(header)
        for i in range(0, len(l)):
            writer.writerow(l[i])


