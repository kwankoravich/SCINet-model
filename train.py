import os
from datetime import datetime
import pandas as pd

# Create output directory
output_dir = os.path.join('saved_models_with_logs', 'stocks', datetime.now().strftime('%HH%MM %d-%b-%Y'))
os.makedirs(output_dir, exist_ok=True)

log_dir = os.path.join(output_dir, 'logs')
diagram_path = os.path.join(output_dir, 'model_diagram.png')

# Proceed with SCINet
model = make_simple_stacked_scinet(X_train.shape, horizon=horizon, K=K, L=L, h=h, kernel_size=kernel_size,
                                   learning_rate=learning_rate, kernel_regularizer=kernel_regularizer,
                                   diagram_path=diagram_path)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0, verbose=1, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=500,
                    callbacks=[early_stopping, tensorboard_callback])

# Save model and training history
model.save(output_dir)
pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'train_history.csv'))