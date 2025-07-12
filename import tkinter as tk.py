import tkinter as tk
from joblib import load
from tkinter import messagebox

model = load("iris_model.joblib")

def predict():
    try:
        features = [
            float(entry_sepal_length.get()),
            float(entry_sepal_width.get()),
            float(entry_petal_length.get()),
            float(entry_petal_width.get())
        ]
        prediction = model.predict([features])[0]
        messagebox.showinfo("Prediction", f"🌸 Iris species: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Iris Species Predictor 🌼")

tk.Label(root, text="Sepal Length").pack()
entry_sepal_length = tk.Entry(root)
entry_sepal_length.pack()

tk.Label(root, text="Sepal Width").pack()
entry_sepal_width = tk.Entry(root)
entry_sepal_width.pack()

tk.Label(root, text="Petal Length").pack()
entry_petal_length = tk.Entry(root)
entry_petal_length.pack()

tk.Label(root, text="Petal Width").pack()
entry_petal_width = tk.Entry(root)
entry_petal_width.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=10)

root.mainloop()
