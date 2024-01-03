"""Data_science

Usage:
    data_science-cli.py train <model-file>
    data_science-cli.py ask <model-file> <input_1> <input_2> <input_3> <input_4> <input_5> <input_6> <input_7> <input_8> <input_9> <input_10> <input_11> <input_12> <input_13> <input_14> <input_15> <input_16>
    data_science-cli.py (-h | --help)
Arguments:
    <model-file>    Serialized model file.
    <input_1>       Age
    <input_2>       Gender (Type 1 for Male and 0 for Female)
    <input_3>       Polyuria (type 1 for Yes and 0 for No)
    <input_4>       Polydipsia (type 1 for Yes and 0 for No)
    <input_5>       Sudden weight loss (type 1 for Yes and 0 for No)
    <input_6>       Weakness (type 1 for Yes and 0 for No)
    <input_7>       Polyphagia (type 1 for Yes and 0 for No)
    <input_8>       Genital thrush (type 1 for Yes and 0 for No)
    <input_9>       Visual blurring (type 1 for Yes and 0 for No)
    <input_10>      Itching (type 1 for Yes and 0 for No)
    <input_11>      Irritability (type 1 for Yes and 0 for No)
    <input_12>      Delayed healing (type 1 for Yes and 0 for No)
    <input_13>      Partial paresis (type 1 for Yes and 0 for No)
    <input_14>      Muscle stiffness (type 1 for Yes and 0 for No)
    <input_15>      Alopecia (type 1 for Yes and 0 for No)
    <input_16>      Obesity (type 1 for Yes and 0 for No)

Options:
    -h --help                  Show this screen.

"""


from data_science import Dataset, SimpleModel
from sklearn.metrics import classification_report
from docopt import docopt

def train_model(model_file):
    dataset = Dataset()
    X_train, y_train = dataset.get_train_set()

    model = SimpleModel()
    model.train(X_train, y_train)

    model.serialize(model_file)

    X_test, y_test = dataset.get_test_set()
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def ask_model(model_file, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16):
    model = SimpleModel().deserialize(model_file)

    y_pred = model.predict([[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, input_12, input_13, input_14, input_15, input_16]] )
    print('\n',y_pred[0])


def main():

    arguments = docopt(__doc__)

    if arguments['train']:
        train_model(arguments['<model-file>'])

    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<input_1>'],
                  arguments['<input_2>'],
                  arguments['<input_3>'],
                  arguments['<input_4>'],
                  arguments['<input_5>'],
                  arguments['<input_6>'],
                  arguments['<input_7>'],
                  arguments['<input_8>'],
                  arguments['<input_9>'],
                  arguments['<input_10>'],
                  arguments['<input_11>'],
                  arguments['<input_12>'],
                  arguments['<input_13>'],
                  arguments['<input_14>'],
                  arguments['<input_15>'],
                  arguments['<input_16>'])


if __name__ == '__main__':
    main()   
