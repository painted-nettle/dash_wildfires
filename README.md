# dash_wildfires
A repository for a group project on the BALADRIA 2025 Summer School. Please note that the data this repository contains are fake.

The repository contains the data and code necessary to generate the distribution data (gen_dists.py), run sentiment analysis (get_sents.py) and topic modelling (get_topics.py) and to then visualise it using Streamlit (app.py).

To run the code:

1. Clone/download the repository, making it a folder on your computer.
2. Install Python (if you do not already have it)
3. Go to the folder (either in a file browser or command line) and create a virtual environment. (_python -m venv venv_ from the command line)
4. Install the required packages from the requirements.txt file. This step is crucial. The rest of the steps won't work without it!
5. Run gen_dists.py (_python gen_dists.py_ from the command line)
6. Run get_sents.py (_python get_sents.py_ from the command line)
7. Run get_topics.py (_python get_topics.py_ from the command line)
8. To see the results in Streamlit, open the command line in the folder and run: _streamlit run app.py_.
