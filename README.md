# 📄 Paragraph to Questions Generator

This is a Gradio-based web application that generates questions from a provided paragraph. The model processes the text, tokenizes the sentences, and produces relevant questions based on the content. This is a useful tool for educators, students, and content creators who need to quickly generate questions from any given text.

## ✨ Features

- 📝 **Text Analysis:** Generates questions from paragraphs by analyzing sentence structure.
- 🌐 **Named Entity Recognition:** Recognizes names, locations, and organizations to ask relevant questions.
- 🔄 **Fallback Mechanism:** If the default tokenizer fails, a fallback method ensures the process continues smoothly.
- 🎯 **Question Generation:** Generates questions from different sentence types like statements, dates, and named entities.
- 🧠 **Robust Handling:** Handles a variety of text structures and content types, including complex and simple sentences.

## 🚀 Example Use Case

- **Input:** A paragraph on a historical event.
- **Output:** Questions such as:
  - "Who was the leader during the event?"
  - "What happened in 1945?"
  - "Why did the event occur?"

## 🎨 Interface Description

- **Input:** A paragraph in the provided text box.
- **Output:** A list of questions generated from the content of the paragraph.

## 🛠️ Technologies Used

- **Gradio:** A Python library to quickly create UIs for machine learning models. Used here to build the interactive interface.
- **NLTK (Natural Language Toolkit):** A library for working with human language data, used for sentence tokenization and part-of-speech tagging.
- **Regular Expressions (Regex):** Used to handle patterns and structure sentences for question generation.
- **Named Entity Recognition (NER):** For recognizing entities such as people, locations, and organizations in the text.
