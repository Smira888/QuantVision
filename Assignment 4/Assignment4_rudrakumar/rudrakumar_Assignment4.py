import os
import numpy as numpy
import pandas as pandas
import yfinance as yfinance
import mplfinance as mplfinance
import tensorflow as tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as pyplot
import seaborn as seaborn

imagesize = 64
windowdays = 20
batchsize = 32
epochcount = 15
imagefolder = "myprojectimages"
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'INTC']
startdate = '2020-01-01'
enddate = '2024-01-01'

def findpattern(row):
    bodysize = abs(row['Open'] - row['Close'])
    lowershadow = min(row['Open'], row['Close']) - row['Low']
    uppershadow = row['High'] - max(row['Open'], row['Close'])
    totalrange = row['High'] - row['Low']
    
    if totalrange == 0:
        return 'None'

    if bodysize <= 0.1 * totalrange:
        return 'Doji'
    
    if (lowershadow > 2 * bodysize) and (uppershadow < 0.5 * bodysize):
        return 'Hammer'
    
    return 'None'

# data lene wala part data lega aur image banaega

def getdataandimages():
    if not os.path.exists(imagefolder):
        os.makedirs(imagefolder)
    
    for foldername in ['Hammer', 'Doji', 'None']:
        fullpath = os.path.join(imagefolder, foldername)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

    alldata = yfinance.download(stocks, start=startdate, end=enddate, group_by='ticker', progress=False)
    
    imagelist = []
    
    mystyle = mplfinance.make_mpf_style(base_mpf_style='yahoo', gridstyle='')

    for symbol in stocks:
        try:
            stockdata = alldata[symbol].dropna().copy()
        except:
            continue

        stockdata['Label'] = stockdata.apply(findpattern, axis=1)

        for i in range(windowdays, len(stockdata)):
            thelabel = stockdata.iloc[i]['Label']
            
            if thelabel == 'None':
                randomnum = numpy.random.rand()
                if randomnum > 0.15: 
                    continue

            filename = f"{symbol}_{i}_{thelabel}.png"
            savepath = os.path.join(imagefolder, thelabel, filename)
            
            if not os.path.exists(savepath):
                datasubset = stockdata.iloc[i-windowdays:i+1]
                try:

                    mplfinance.plot(datasubset, type='candle', style=mystyle, 
                                    axisoff=True, volume=False, savefig=savepath, 
                                    figsize=(2, 2), closefig=True)
                except:
                    pass
            
            imagelist.append({'path': savepath, 'label': thelabel})

    return pandas.DataFrame(imagelist)

mydataframe = getdataandimages()

# taste data split training wala part

traindata, testdata = train_test_split(mydataframe, test_size=0.2, stratify=mydataframe['label'], random_state=42)

mygenerator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

traingen = mygenerator.flow_from_dataframe(
    dataframe=traindata, x_col='path', y_col='label',
    target_size=(imagesize, imagesize), batch_size=batchsize,
    class_mode='categorical', subset='training', shuffle=True
)

valgen = mygenerator.flow_from_dataframe(
    dataframe=traindata, x_col='path', y_col='label',
    target_size=(imagesize, imagesize), batch_size=batchsize,
    class_mode='categorical', subset='validation', shuffle=True
)


mymodel = models.Sequential()

mymodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imagesize, imagesize, 3)))
mymodel.add(layers.MaxPooling2D((2, 2)))

mymodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
mymodel.add(layers.MaxPooling2D((2, 2)))

mymodel.add(layers.Flatten())
mymodel.add(layers.Dense(64, activation='relu'))
mymodel.add(layers.Dropout(0.5))
mymodel.add(layers.Dense(3, activation='softmax'))

mymodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = mymodel.fit(traingen, epochs=epochcount, validation_data=valgen)

# plating wala part

testgen = mygenerator.flow_from_dataframe(
    dataframe=testdata, x_col='path', y_col='label',
    target_size=(imagesize, imagesize), batch_size=batchsize,
    class_mode='categorical', shuffle=False
)

predictions = mymodel.predict(testgen)
predclasses = numpy.argmax(predictions, axis=1)
trueclasses = testgen.classes
classnames = list(testgen.class_indices.keys())

seaborn.set_style("whitegrid")

fig, axes = pyplot.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance Metrics', fontsize=16, weight='bold')


axes[0].plot(history.history['accuracy'], label='Training Acc', color='blue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Acc', color='orange', linewidth=2)
axes[0].set_title('Accuracy', fontsize=12)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
axes[1].set_title('Loss', fontsize=12)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()

matrixdata = confusion_matrix(trueclasses, predclasses)
seaborn.heatmap(matrixdata, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classnames, yticklabels=classnames, ax=axes[2], cbar=False)
axes[2].set_title('Confusion Matrix', fontsize=12)
axes[2].set_ylabel('True Label')
axes[2].set_xlabel('Predicted Label')

pyplot.tight_layout()
pyplot.show()

print(classification_report(trueclasses, predclasses, target_names=classnames))