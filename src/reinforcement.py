import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import random
from collections import deque
import time
import IPython.display

# Define the Reinforcement Learning agent
class DeepQLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) 
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural network for Deep Q-Learning Model with LSTM
        inputs = Input(shape=(self.state_size[0], self.state_size[1]))
        lstm_layer = LSTM(128, activation='relu', return_sequences=False)(inputs)
        dense_layer = Dense(64, activation='relu')(lstm_layer)
        outputs = Dense(self.action_size, activation='linear')(dense_layer)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        
        # Returning the highest Q-Value
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        


# Function to preprocess data and create sequences for prediction
def preprocess_data_for_prediction(df, n_steps_in, n_steps_out):
    
    # Encode categorical data and scale numerical values
    encoder_weather = LabelEncoder()
    df['Weather_Conditions'] = encoder_weather.fit_transform(df['Weather_Conditions'])
    encoder_tariff = LabelEncoder()
    df['Tariff_Type'] = encoder_tariff.fit_transform(df['Tariff_Type'])
    scaler = MinMaxScaler()
    df[['Grid_Load (MW)', 'Tariff_Rate']] = scaler.fit_transform(df[['Grid_Load (MW)', 'Tariff_Rate']])

    # Converting to numpy array and create sequences for fast processing
    dataset = df[['Weather_Conditions', 'Grid_Load (MW)', 'Tariff_Type', 'Tariff_Rate']].values
    X, y = [], []
    for i in range(len(dataset) - n_steps_in - n_steps_out):
        X.append(dataset[i:i + n_steps_in, :])
        y.append(dataset[i + n_steps_in:i + n_steps_in + n_steps_out, -1])  # Predicting 'Tariff_Rate'

    return np.array(X), np.array(y)

# Loading and preprocessing the dataset
df = pd.read_csv('../dataset/TOU_Tariffs_Dataset_5000.csv')
df = df.drop(columns=['Tariff_ID', 'TOU_Period_Start', 'TOU_Period_End', 'Forecasted_Tariff_Rate', 'Renewable_Energy_Contribution (%)'], axis=1)

# Defining sequence lengths
n_steps_in, n_steps_out = 48, 24

# Preparing data for training and testing
X, y = preprocess_data_for_prediction(df, n_steps_in, n_steps_out)

# Split the data into training and testing sets Reshape y_train and y_test for LSTM model compatibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

# Training the Deep Q-Learning agent
agent = DeepQLearningAgent(state_size=(n_steps_in, X.shape[2]), action_size=n_steps_out)

# Defining the hyperparameters
episodes = 50
batch_size = 32

# Training loop for the agent
for e in range(episodes):
    idx = np.random.randint(0, X_train.shape[0] - 1)
    state = X_train[idx].reshape(1, n_steps_in, X.shape[2])

    for t in range(n_steps_out):
        action = agent.act(state)
        next_state = X_train[np.random.randint(0, X_train.shape[0] - 1)].reshape(1, n_steps_in, X.shape[2])
        reward = -y_train[idx, action, 0] 
        done = t == n_steps_out - 1
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"Episode: {e + 1}/{episodes}, Score: {reward}, Epsilon: {agent.epsilon}")
            time.sleep(3)
            os.system('cls')
            break

    # Replay training
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
        
# Predicting the tariff
def predict_next_24_hours(agent, X_test):
    predictions = []
    for i in range(X_test.shape[0]):
        state = X_test[i].reshape(1, n_steps_in, X_test.shape[2])
        predicted_sequence = []
        for _ in range(n_steps_out):
            action = agent.act(state)
            predicted_value = action
            predicted_sequence.append(predicted_value)
        predictions.append(predicted_sequence)
        os.system('cls')
    return np.array(predictions)

# Make predictions on the test set
predictions = predict_next_24_hours(agent, X_test)

# Inverse transform to get the actual tariff values (scaling)
scaler = MinMaxScaler()
predicted_tariffs = scaler.inverse_transform(predictions)

# Evaluate the model
print(f"Predicted values for next 24 hours: {predicted_tariffs}")