#Liberaries
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from keras.models import Sequential , load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import os

"""
Image Classfier for Multiclass problem on tuberculosis data classiying 'Normal', 'Cavitry' and 'Millary'
Use Augment Data Notebook to Augment Data First

"""
class ImageClassification(object):


    def __init__(self):

        self.IMAGE_WIDTH=128
        self.IMAGE_HEIGHT=128
        self.IMAGE_SIZE=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.IMAGE_CHANNELS=3
        self.batch_size=10

    # create dataframe for train/test data
    def loadData(self, datasetPath):

        filenames = os.listdir(datasetPath)
        categories = []
        for filename in filenames:
            category = filename.split(" ")[0]
            if category == 'Normal':
                categories.append('Normal')
            elif category == 'Cavitry':
                categories.append('Cavitry')
            elif category == "Millary":
                categories.append('Millary')

        df = pd.DataFrame({
            'filename': filenames,
            'category': categories
        })
        df = df.reset_index(drop = True)

        return df

    def splitTrainValidationData(self, dataFrame):

        train_df, val_df = train_test_split(dataFrame, test_size=0.20, random_state=42, stratify= dataFrame[['category']])

        train_df = train_df.reset_index(drop=True)
        val_df  = val_df.reset_index(drop=True)

        return train_df, val_df

    #prepare Image generators for train and validation data
    def prepareImageGenerator(self, df, data_path):

        datagen = ImageDataGenerator(
        rotation_range=10,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        )

        data_generator = datagen.flow_from_dataframe(
            df,
            data_path,
            x_col='filename',
            y_col='category',
            target_size= self.IMAGE_SIZE,
            class_mode='categorical',
            batch_size= self.batch_size
        )

        return data_generator



    #Creates and Return Deep Convlutional Neural Network For Image classification
    def create_model (self):

        kernel_size = (3,3)
        pool_size= (2,2)
        first_filters = 32
        second_filters = 64
        third_filters = 128
        fourth_filters = 256

        dropout_conv = 0.1

        dropout_dense = 0.2


        model = Sequential()
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)))
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pool_size)) 
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
        model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
        model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
        model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
        model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Flatten())
        model.add(Dense(fourth_filters, activation = "relu"))
        model.add(Dropout(dropout_dense))
        model.add(Dense(3, activation = "softmax"))  # since we have 3 classes, normal, cavitry and millary

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        return model

    #Trains Model
    def trainModel(self, model, train_generator, validation_generator , no_epochs ,bestModelPath):

        # train and validation steps for train and validation
        train_steps = np.ceil(train_generator.samples / self.batch_size)
        val_steps = np.ceil(validation_generator.samples / self.batch_size)


        model_checkpoint = ModelCheckpoint('CIFAR10{epoch:02d}.h5', save_weights_only=True,
                                           save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [model_checkpoint]
        history = model.fit_generator(
            train_generator,
            epochs=no_epochs,
            validation_data=validation_generator,
            validation_steps=val_steps,
            steps_per_epoch=train_steps,
            verbose=1,
            shuffle=True,
            callbacks=callbacks

        )

        model.save_weights(filepath= bestModelPath)  #Saves best Model weights


    #Load Model with best weights
    def load_model(self, model, bestModelPath):

        model.load_weights(bestModelPath)
        return model

        # Creates Image Generators for test data

    def prepareTestGenerator(self, df, data_path):

        dataGen = ImageDataGenerator(
            rescale=1. / 255,
        )

        test_generator = dataGen.flow_from_dataframe(
            df,
            data_path,
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )
        return test_generator

    #Evaluate model on test data and prepate results report
    def evaluateModelOnTestData(self, model, testDatagenerator):
        
        test_loss, test_acc = model.evaluate_generator(testDatagenerator)
        print('test_loss:', test_loss)
        print('test_acc:', test_acc)

        target_names = np.empty([1, 3],dtype=object)[0]
        for key , value in testDatagenerator.class_indices.items():
            target_names[value] =  key
        target_names = list(target_names)

        #Confusion Matrix

        Y_pred = model.predict_generator(testDatagenerator)
        y_pred = np.argmax(Y_pred, axis=1)
        cnf_matrix = confusion_matrix(testDatagenerator.classes, y_pred)
        conf_mat = pd.DataFrame(cnf_matrix, columns = target_names, index= target_names)
        print('\n*******************Confusion Matrix****************\n')
        print(conf_mat)
        print('\n*************************************************\n')

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # true positive rate
        TPR = TP/(TP+FN)
        # false poitive rate
        FPR = FP/(FP+TN) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        TNR = TN/(TN+FN)
        # False negative rate
        FNR = FN/(TP+FN)
        ERR = (FP+FN)/(TP+FP+FN+TN)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        Multiclass_Evaluation_Matrix = pd.DataFrame([FP,FN,TP,TN, TPR , FPR , TNR,FNR,ERR,  ACC], columns = target_names, index= ['False Positive', 'Fase Negitive', 'True Positive', 'True Negetive', 'True Positive Rate', 'False Positive Rate', 'True Negetiev Rate','False Negetive rate', 'Error Rate','Accuracy'])
        print('\n*****************Multiclass Evaluation Matrix******************\n')
        print(Multiclass_Evaluation_Matrix)
        print('\n*************************************************\n')
        

        print('\n*************Classification Report***************\n')

        print(classification_report(testDatagenerator.classes, y_pred, target_names=target_names))
        print('\n*************************************************\n')




#Driver Code
if __name__ == "__main__":


    # path for train data
    dataset_path_train = r"C:\Users\wel\Desktop\Image Classification\Data Set\Train\trainData"
    # Path for test data
    dataset_path_test = r"C:\Users\wel\Desktop\Image Classification\Data Set\Test\testData"
    #Path to save best model weights
    bestModelPath = r"C:\Users\wel\Desktop\Image Classification\final_weight.h5"

    #class Object
    imgClassifier = ImageClassification()


    #Load train and Test data
    trainData = imgClassifier.loadData(dataset_path_train)
    testData = imgClassifier.loadData(dataset_path_test)

    #Splits Train and Validation Data
    train_df, val_df = imgClassifier.splitTrainValidationData(trainData)

    #Prepares Image generators for train and validation data
    trainDataGenerator = imgClassifier.prepareImageGenerator(train_df, dataset_path_train)
    validationDataGenerator = imgClassifier.prepareImageGenerator(val_df, dataset_path_train)

    #Get Model
    model = imgClassifier.create_model()

    #Train Model and Save best Model Weights
    imgClassifier.trainModel( model, trainDataGenerator, validationDataGenerator, 25 , bestModelPath)

    #Get Model for test data
    testModel = imgClassifier.create_model()

    #Load best weights for model
    testModel = imgClassifier.load_model(testModel,bestModelPath)


    #Prepare Image Generator for test data
    testDataGenerator = imgClassifier.prepareTestGenerator(testData, dataset_path_test)

    #Evaluate Model on test data
    imgClassifier.evaluateModelOnTestData(testModel, testDataGenerator)





