import streamlit as st
from openai import OpenAI
import psycopg2
import os
import json
from Modules.article import VitoArticle
from Modules.label_assignment import assign_labels_a_star
from Modules.utils import parse_pg_array, UploadOpenAI
from Modules.data_processing import clean_html

# Set up your OpenAI API key
openaiClient = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization= os.getenv("OPENAPI_ORG")
)

# Title of the app
st.title("VITO Label Assignment")

st.session_state['List_result_labels'] = []

# Sidebar for navigation
options = ["Assign Labels"]
choice = st.sidebar.radio("Go to", options)

if choice == "Assign Labels":
    st.header("Assign Labels to an Article")

    # Upload article content
    uploaded_file = st.file_uploader("Upload an article JSON file", type=["json"])
    if uploaded_file is not None:
        # Clean list
        st.session_state['List_result_labels'] = []
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create a VitoArticle instance
        article = VitoArticle(temp_file_path)

        with st.expander('Uploaded article'):
            st.write("Uploaded internal id file:", article.artikel_id)
            st.write("Uploaded article name:", article.artikel)
            st.write(clean_html(article.inhoud))

        # Check if the article content is invalid
        if article.inhoud == "<div>\r\n     <div>[...]</div>\r\n   </div>" or article.inhoud == "<p>...</p>" or article.inhoud == "<p>[...]</p>": 
            st.warning("Skipping due to invalid HTML/article")
        else:
            try:
                # Generate embedding
                with st.spinner("Generating embedding..."):
                    article.embedding = article.get_embedding(openaiClient)

                # Extract keywords
                with st.spinner("Extracting keywords..."):
                    keyword_result = UploadOpenAI(article.inhoud, openaiClient)
                    # Parse the response to get keywords
                    try:
                        article.add_keywords([keyword['Keyword'] for keyword in json.loads(keyword_result.choices[0].message.content)["metadata"]])
                    except json.JSONDecodeError:
                        st.error("Failed to parse keywords from OpenAI response.")
                        st.stop()

                # Assign labels
                with st.spinner("Assigning labels..."):
                    # Get database connection
                    conn = psycopg2.connect(
                        host="localhost",
                        database="postgres",
                        user="postgres",
                        password="root",
                        port="5432"
                    )
                    result_labels, assigned_labels, outputs = assign_labels_a_star(article, conn)
                    conn.close()

                # Collect results for evaluation
                st.session_state['List_result_labels'].append({
                    "Artikel": article.artikel_id,
                    "Artikel_name": article.artikel,
                    "List_result_labels": result_labels,
                    "correct_labels": article.labels
                })

                # Display the assigned labels
                st.subheader("Assigned Labels")
                for labels in result_labels:
                    st.write(" > ".join(labels))

                # Optionally, display the correct labels for comparison
                st.subheader("Correct Labels")
                st.write(" > ".join(article.labels))

                # Display the processing details
                st.subheader("Processing Details")

                # Organize outputs using containers
                for output in outputs:
                    if isinstance(output, dict):
                        output_type = output.get('type')
                        if output_type == 'label_mean_score':
                            st.write(f"Label: {output['label']}, Mean Score: {output['mean_score']:.4f}")
                        elif output_type == 'label_scores':
                            with st.expander(f"Label Scores at Level {output['level']}"):
                                for label, scores in output['label_scores'].items():
                                    st.write(f"Label: {label}, Scores: {scores}")
                        elif output_type == 'requery':
                            st.write(f"Re-querying with label path: {output['label_path']}")
                        elif output_type == 'assigned_label':
                            st.write(f"Assigned Label for top mean score: {output['top_mean_score']}")
                        elif output_type == 'final_labels':
                            st.subheader("Assigned Labels with Mean Scores:")
                            for item in output['assigned_labels']:
                                st.write(f"Labels: {' > '.join(item['path'])}, Mean Score: {item['mean_score']:.4f}")
                        elif output_type == 'info':
                            st.write(output['message'])
                    else:
                        # Output from get_overlapping_keywords_info
                        article_info = output
                        with st.container():
                            st.markdown(f"### {article_info['stage']}")
                            st.write(f"**Article ID:** {article_info['article_id']}")
                            st.write(f"**Labels:** {article_info['labels']}")
                            st.write(f"**Semantic Similarity:** {article_info['semantic_similarity']}")
                            st.write(f"**Combined Score:** {article_info['combined_score']}")
                            st.write(f"**Test Article Keywords:** {article_info['test_keywords']}")
                            st.write(f"**Article Keywords:** {article_info['article_keywords']}")

                            if article_info['overlapping_keywords']:
                                st.write("**Overlapping Keywords:**")
                                for overlap in article_info['overlapping_keywords']:
                                    st.write(f"- Test Keyword: '{overlap['test_keyword']}', "
                                             f"Article Keyword: '{overlap['article_keyword']}', "
                                             f"Article Keyword Weight: {overlap['article_weight']}")
                            else:
                                st.write("No overlapping keywords.")

            except Exception as e:
                st.error(f"An error occurred while processing the article: {e}")