import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import zipfile
from io import BytesIO
import pickle

st.set_page_config(page_title="Epsilon", page_icon='🤖', menu_items={
    'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]"
})

# Initialize session state keys
if 'labels' not in st.session_state:
    st.session_state['labels'] = {}
if 'num_classes' not in st.session_state:
    st.session_state['num_classes'] = 0
if 'label_mapping' not in st.session_state:
    st.session_state['label_mapping'] = {}
if 'model' not in st.session_state:
    st.session_state['model'] = None

# Function to train the model with progress
def train_model(images, labels, num_classes, epochs, progress_bar):
    X = np.array(images)
    y = np.array(labels)
    X = X / 255.0
    y = to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
        progress_bar.progress((epoch + 1) / epochs)

    return model

# Function to save the model in the specified format
def save_model(model, export_format, usage_code):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if export_format == 'tflite':
            input_shape = (1, 64, 64, 3)
            run_model = tf.function(lambda x: model(x))
            concrete_func = run_model.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            tflite_model = converter.convert()
            zf.writestr("model.tflite", tflite_model)
        elif export_format == 'h5':
            model.save("model.h5")
            zf.write("model.h5")

        zf.writestr("main.py", usage_code)

    buffer.seek(0)
    return buffer

# Function to import the .epsilon file (same as .zip)
def import_from_epsilon(epsilon_file):
    with zipfile.ZipFile(epsilon_file, 'r') as zf:
        # Load labels
        labels_data = zf.read("labels.pkl")
        st.session_state['labels'] = pickle.loads(labels_data)

        # Load images
        for filename in zf.namelist():
            if filename.endswith(".png"):
                label = filename.split("_")[0]
                img_data = zf.read(filename)
                img_array = image.img_to_array(image.load_img(BytesIO(img_data)))
                if label in st.session_state['labels']:
                    st.session_state['labels'][label].append(img_array)

        st.session_state['num_classes'] = len(st.session_state['labels'])
        st.success("EPSILON file imported successfully!")

# Function to test the model with a new image
def test_model(model, img_array, label_mapping):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    confidence = np.max(prediction)

    labels_reverse_map = {v: k for k, v in label_mapping.items()}
    predicted_label = labels_reverse_map[predicted_label_index]
    return predicted_label, confidence

# Streamlit app
st.title(":red[Epsilon (Model Creator)]")

# Sidebar for label input
st.sidebar.title(":blue[Manage Labels]")
if 'label_input' not in st.session_state:
    st.session_state.label_input = ""

label_name = st.sidebar.text_input("Enter a new label:", value=st.session_state.label_input)
if st.sidebar.button("Add Label"):
    if label_name and label_name not in st.session_state['labels']:
        st.session_state['labels'][label_name] = []
        st.session_state['num_classes'] += 1
        st.sidebar.success(f"Label '{label_name}' added!")
        st.session_state.label_input = ""
    else:
        st.sidebar.warning("Label already exists or is empty.")

# Dropdown to select model export format
export_format = st.sidebar.selectbox("Select model export format:", options=["tflite", "h5"])

# Display the existing labels and allow image upload in rows
if st.session_state['num_classes'] > 0:
    num_columns = 3
    cols = st.columns(num_columns)

    for i, label in enumerate(st.session_state['labels']):
        with cols[i % num_columns]:
            st.subheader(f"Upload images for label: {label}")
            uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=label)

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    image_data = image.load_img(uploaded_file, target_size=(64, 64))
                    image_array = image.img_to_array(image_data)
                    st.session_state['labels'][label].append(image_array)
                st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

# Advanced options in sidebar
with st.sidebar.expander("Advanced Options"):
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)

# Button to train the model
if st.session_state['num_classes'] > 1:
    if st.button("Train Model"):
        all_images = []
        all_labels = []
        st.session_state['label_mapping'] = {label: idx for idx, label in enumerate(st.session_state['labels'].keys())}

        for label, images in st.session_state['labels'].items():
            all_images.extend(images)
            all_labels.extend([st.session_state['label_mapping'][label]] * len(images))

        if len(all_images) > 0:
            st.write("Training the model...")
            progress_bar = st.progress(0)
            st.session_state['model'] = train_model(all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar)
            st.success("Model trained!")
        else:
            st.error("Please upload some images before training.")
else:
    st.warning("At least two labels are required to train the model.")

# Option to test the trained model
if st.session_state['model'] is not None:
    st.subheader("Test the trained model with a new image")
    test_image = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png'], key="test")

    if test_image:
        test_image_data = image.load_img(test_image, target_size=(64, 64))
        st.image(test_image_data, caption="Uploaded Image", use_column_width=True)

        test_image_array = image.img_to_array(test_image_data)
        predicted_label, confidence = test_model(st.session_state['model'], test_image_array, st.session_state['label_mapping'])

        st.write(f"Predicted Label: {predicted_label}")
        st.slider("Confidence Level (%)", min_value=1, max_value=100, value=int(confidence * 100), disabled=True)

# Initialize usage_code
usage_code = ""

# Button to download the model
if st.session_state['model'] is not None and st.button("Download Model"):
    try:
        predicted_label_code = ', '.join([f"'{k}': {v}" for k, v in st.session_state['label_mapping'].items()])

        if export_format == 'tflite':
            usage_code = f"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Load and preprocess the image
img = image.load_img('path/to/your/image.jpg', target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make a prediction
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

# Get the predicted label
predicted_label_index = np.argmax(predictions)
labels = {{{predicted_label_code}}}
predicted_label = labels[predicted_label_index]
print(f"Predicted Label: {{predicted_label}}")
"""
        elif export_format == 'h5':
            usage_code = f"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('model.h5')

# Load and preprocess the image
img = image.load_img('path/to/your/image.jpg', target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make a prediction
predictions = model.predict(img_array)

# Get the predicted label
predicted_label_index = np.argmax(predictions)
labels = {{{predicted_label_code}}}
predicted_label = labels[predicted_label_index]
print(f"Predicted Label: {{predicted_label}}")
"""

        model_buffer = save_model(st.session_state['model'], export_format, usage_code)
        st.download_button(f"Download {export_format.upper()} Model", model_buffer, f"model.{export_format}.zip", mime="application/zip")
    except Exception as e:
        st.error(f"Error exporting model: {e}")

# Function to export to .epsilon format
if st.sidebar.button("Export to .epsilon"):
    try:
        epsilon_buffer = BytesIO()
        with zipfile.ZipFile(epsilon_buffer, "w") as zf:
            zf.writestr("labels.pkl", pickle.dumps(st.session_state['labels']))
            for label, images in st.session_state['labels'].items():
                for i, img_array in enumerate(images):
                    img = image.array_to_img(img_array)
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    zf.writestr(f"{label}_{i}.png", img_byte_arr.getvalue())

        epsilon_buffer.seek(0)
        st.download_button("Download .epsilon file", epsilon_buffer, "model.epsilon", mime="application/zip")
    except Exception as e:
        st.error(f"Error exporting to .epsilon: {e}")

# Button to import the .epsilon file
import_file = st.sidebar.file_uploader("Import .epsilon file", type=['epsilon', 'zip'])

if import_file is not None:
    try:
        import_from_epsilon(import_file)
    except Exception as e:
        st.error(f"Error importing .epsilon file: {e}")
