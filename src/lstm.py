import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.keras',  # Save with epoch number in the filename
    save_freq='epoch',  # Saves based on epochs
    save_weights_only=False,  # Save the full model (architecture + weights)
    verbose=1  # Show message after each save
)

# Custom callback for saving after every 5 epochs
class CustomSaveCallback(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            super().on_epoch_end(epoch, logs)

# Use the custom callback in model.fit()
custom_checkpoint_callback = CustomSaveCallback(
    filepath='model_epoch_{epoch:02d}.keras',  # Save with epoch number in the filename
    save_weights_only=False,  # Save the full model
    verbose=1
)

# Load the dataset
df = pd.read_csv('..dataset/TOU_Tariffs_Dataset_5000.csv')

df = df.drop(columns=['Tariff_ID', 'TOU_Period_Start', 'TOU_Period_End', 'Forecasted_Tariff_Rate', 'Renewable_Energy_Contribution (%)'], axis=1)

# Preprocess the data
# Encode 'weather_condition' and 'tariff_type' as these are categorical features
encoder_weather = LabelEncoder()
df['Weather_Conditions'] = encoder_weather.fit_transform(df['Weather_Conditions'])

encoder_tariff = LabelEncoder()
df['Tariff_Type'] = encoder_tariff.fit_transform(df['Tariff_Type'])

# Scale numerical columns
scaler = MinMaxScaler()
df[['Grid_Load (MW)', 'Tariff_Rate']] = scaler.fit_transform(df[['Grid_Load (MW)', 'Tariff_Rate']])

# Prepare the dataset for LSTM
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        # Find the end of the current sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        if out_end_ix > len(data):
            break
        
        # Input and output sequences
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

# Define input/output sequence lengths
n_steps_in, n_steps_out = 48, 24  # Input 48 hours to predict the next 24

# Convert dataframe to numpy array
dataset = df[['Weather_Conditions', 'Grid_Load (MW)', 'Tariff_Type', 'Tariff_Rate']].values

# Create sequences
X, y = create_sequences(dataset, n_steps_in, n_steps_out)

# Reshape y to 3D
y = y.reshape((y.shape[0], y.shape[1], 1))

### 2. Building the LSTM Encoder-Decoder Model
# Encoder-Decoder Model

n_features = X.shape[2]  # Number of features per timestep

# Encoder
encoder_inputs = Input(shape=(n_steps_in, n_features))
encoder_lstm = LSTM(128, activation='relu', return_sequences=False)(encoder_inputs)
encoder_output = RepeatVector(n_steps_out)(encoder_lstm)

# Decoder
decoder_lstm = LSTM(128, activation='relu', return_sequences=True)(encoder_output)
decoder_output = TimeDistributed(Dense(1))(decoder_lstm)

# Define the model
model = Model(inputs=encoder_inputs, outputs=decoder_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Fit the model
history = model.fit(
    X, y, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[checkpoint_callback]
)