import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from generator.Board import Board
from recognize_sudoku import recognize
from sudoku_solver import read_from_file, all_board_non_zero, solve

old_sudoku, model = None, None
# Loading model (Load weights and configuration seperately to speed up model.predict)
input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Load weights from pre-trained model. This model is trained in digitRecognition.py
model.load_weights("digitRecognition.h5")
# detector_model, recognizer_model = load_model()
st.set_page_config(
    page_title="Sudoku Solver App",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="expanded", )
st.title("Sudoku Solver")
import time

with st.empty():
   for seconds in range(7):
       st.write(f"⏳ {seconds} solving dependencies...")
       time.sleep(1)
       st.write("✔️ Done! Now, you can upload image!")
st.sidebar.markdown(
    """
    <h1  style="color:blue;">Sudoku Solver</h1>
    <p>
    <h3 style="color:green;">Hello I am your helper!🕵️‍♂️</h3></br>
    This a Sudoku Solver app that uses a custom OCR to detect digits in a cropped screenshot of a sudoku grid and then
    using simple recursive algorithm to solve it before displaying the results.
    </p>
    <p style="color:red;">
    Upload an image of a sudoku grid(Use PNG file if possible) and get the solved state.
    </p>
    <b>Note:</b>
    <ul><li>Please upload some high quality image!</li><li>Please upload some other image if error occured!🙏</ul>
    <img src='https://raw.githubusercontent.com/CVxTz/sudoku_solver/master/solver/samples/wiki_sudoku.png'
         width="300"></br>
    Image source : <a href='https://en.wikipedia.org/wiki/Sudoku'>Wikipedia Sudoku</a> </br>
    <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
    Made by : <a href='https://www.linkedin.com/in/yashmodi59/'>Yash Modi</a> </br>
    """,
    unsafe_allow_html=True,
)
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
file = st.file_uploader("Upload Sudoku image", type=["jpg", "png"])

if file:
    img = read_from_file(file)
    start = time.time()
    grid, image = recognize(img, model, old_sudoku)
    solve_duration = time.time() - start
    if not (all_board_non_zero(grid)):
        st.error('Could not detect😔 sudoku please try again or try with some other resolution')
    else:
        st.success("Solved!")
        solving_time = st.empty()
        solving_time.markdown(
            "<center>"
            + "<h3>Solved in %.5f seconds </h3>" % solve_duration
            + "</center>",
            unsafe_allow_html=True,
        )
        st.image(image, caption='Solved Sudoku',
                 use_column_width=True)
