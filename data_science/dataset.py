import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset():

    def __init__(self):
        connection = mysql.connector.connect(
            user='admin', password='admin', host='localhost', port=3316, database='daredatachallenge')
        
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM data')
        data = cursor.fetchall()
        connection.close()

        df = pd.DataFrame(columns=list(data[0]))

        for i in range(1, len(data)):
            df.loc[len(df.index)] = list(data[i])

        df.replace(['Male', 'Yes'], 1, inplace=True)
        df.replace(['Female', 'No'], 0, inplace=True)
        df.replace('Positive\r', 'Positive', inplace=True)
        df.replace('Negative\r', 'Negative', inplace=True)
        df['Age']=df['Age'].astype(int)

        self.X = df.iloc[:,:-1]
        self.y = df.iloc[:,-1]

        self.split_train_test()

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)
    
    def get_train_set(self):
        return self.X_train, self.y_train
    
    def get_test_set(self):
        return self.X_test, self.y_test