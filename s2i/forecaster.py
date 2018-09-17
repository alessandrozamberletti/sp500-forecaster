from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn import metrics


class Forecaster(Sequential):
    def __init__(self, trasformer, debug=False, debug_dir='.'):
        super(Sequential, self).__init__()
        self.trasformer = trasformer
        self.debug = debug
        self.debug_dir = debug_dir
        self.__setup_architecture()

    def fit_to_data(self, x, y, epochs=100, validation_split=0.2):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
        self.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        hist = self.fit(x, y, shuffle=True, epochs=epochs, validation_split=validation_split, callbacks=[es])
        if self.debug:
            self.__save_loss(hist)
        return self

    def evaluate(self, x, y, ohlcv, ticker):
        preds = self.predict_classes(x)

        plt.cla()

        # no data before timestep and after -futurestep
        chart = ohlcv[self.trasformer.features[-1]].values[self.trasformer.timestep:-self.trasformer.futurestep]
        plt.plot(chart, color='black')

        # gt vs predicted
        for idx, (gt, pred) in enumerate(zip(y, preds)):
            if gt != pred:
                plt.axvspan(idx, idx + 1, fc='gray')
                continue
            plt.axvspan(idx, idx + 1, fc='green', alpha=.5) if pred else plt.axvspan(idx, idx + 1, fc='red', alpha=.5)

        plt.title('Predictions vs Ground Truth - SYMBOL:{}'.format(ticker))
        plt.ylabel('{} price'.format(self.trasformer.features[-1]))
        plt.xlabel('days')

        legend = [mpatches.Patch(label='pos. outlook (gt == pred)', color='green', alpha=.5),
                  mpatches.Patch(label='neg. outlook (gt == pred)', color='red', alpha=.5),
                  mpatches.Patch(label='wrong outlook (gt != pred)', color='gray', alpha=.5)]

        plt.legend(handles=legend)

        self.__save_img(ticker)

        return metrics.accuracy_score(y, preds)

    def __setup_architecture(self):
        self.add(Conv2D(32, (2, 2), input_shape=self.trasformer.time_window_shape))
        self.add(Conv2D(32, (2, 2)))
        self.add(Conv2D(32, (2, 2)))
        self.add(Conv2D(32, (3, 3)))
        self.add(Flatten())
        self.add(Dense(128))
        self.add(Dropout(.2))
        self.add(Dense(256))
        self.add(Dense(units=1, activation='sigmoid'))

    def __save_loss(self, data):
        plt.plot(data.history['loss'], label='train')
        plt.plot(data.history['val_loss'], label='validation')

        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()

        self.__save_img('loss')

    def __save_img(self, im_name):
        fn = os.path.join(self.debug_dir, '{}.png'.format(im_name))
        plt.savefig(fn, dpi=300, bbox_inches='tight')
