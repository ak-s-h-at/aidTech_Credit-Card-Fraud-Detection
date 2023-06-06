import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the saved model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

def process_input():
    # Get the user input
    user_input = input_entry.get()

    # Split the input by comma and convert to float
    values = user_input.split(",")
    values = [float(value.strip()) for value in values]

    # Fill the entry fields with the user input values
    for i, entry in enumerate(entry_widgets):
        entry.delete(0, tk.END)
        entry.insert(tk.END, values[i])

    # Scale the user input
    scaled_user_input = scaler.transform([values])

    # Make predictions
    prediction = model.predict(scaled_user_input)

    # Display the prediction
    messagebox.showinfo("Prediction", f"The prediction is: {'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'}")

# Create the main GUI
window = tk.Tk()
window.title("Credit Card Fraud Detection")

# Create an Entry widget for user input
input_label = tk.Label(window, text="Enter 28 values separated by commas:")
input_label.pack()
input_entry = tk.Entry(window)
input_entry.pack()

# Create a button to process the input
submit_button = tk.Button(window, text="Submit", command=process_input)
submit_button.pack()

# Create Entry widgets for each feature
entry_widgets = []
for i in range(28):
    label = tk.Label(window, text=f"v{i+1}")
    label.pack()
    entry = tk.Entry(window)
    entry.pack()
    entry_widgets.append(entry)

# Run the main GUI
window.mainloop()
