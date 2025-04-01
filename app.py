import nltk
import re
import random
import os
import gradio as gr

# Set NLTK data path to a writable location in Hugging Face environment
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Explicitly download both punkt and punkt_tab resources
def ensure_nltk_resources():
    resources = [
        'punkt', 
        'punkt_tab',  # Add this specific resource that's causing the error
        'averaged_perceptron_tagger', 
        'maxent_ne_chunker', 
        'words'
    ]
    
    for resource in resources:
        try:
            # First check if already downloaded
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"Resource {resource} already downloaded")
            except LookupError:
                print(f"Downloading {resource}...")
                nltk.download(resource, download_dir=nltk_data_path)
                print(f"Downloaded {resource}")
        except Exception as e:
            print(f"Warning: Could not download {resource}: {str(e)}")

# Ensure resources are downloaded before proceeding
print("Setting up NLTK resources...")
ensure_nltk_resources()

# Simple sentence tokenizer as fallback
def simple_sentence_tokenizer(text):
    """A simpler fallback sentence tokenizer."""
    sentences = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if sentence:
            sentences.append(sentence)
    return sentences

def get_named_entities(text):
    """Extract named entities from text with error handling."""
    try:
        from nltk.tag import pos_tag
        from nltk.chunk import ne_chunk
        from nltk.tree import Tree
        
        tokens = nltk.word_tokenize(text)
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged)
        
        named_entities = []
        for chunk in chunked:
            if isinstance(chunk, Tree):
                entity = ' '.join([word for word, tag in chunk.leaves()])
                named_entities.append((entity, chunk.label()))
        
        return named_entities
    except Exception as e:
        print(f"Named entity recognition failed: {str(e)}")
        return []

def generate_question_from_sentence(sentence):
    """Generate a question from a sentence with robust error handling."""
    try:
        # Remove punctuation at the end
        sentence = re.sub(r'[.!?]$', '', sentence)
        
        # Check for common patterns that can be turned into questions
        if re.search(r'\bis\s|\bwas\s|\bwere\s|\bare\s', sentence):
            # Convert statements with "is", "was", "were", "are" into yes/no questions
            match = re.search(r'^(.*?)\s(is|was|were|are)\s(.*?)$', sentence, re.IGNORECASE)
            if match:
                return f"{match.group(2).capitalize()} {match.group(1)} {match.group(3)}?"
        
        # Check for sentences with dates or years
        if re.search(r'\b(in|on|during)\s\d{4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', sentence, re.IGNORECASE):
            return f"When did {sentence.lower()}?"
        
        # Try to get named entities, but don't fail if NER isn't working
        entities = get_named_entities(sentence)
        
        # If there are named entities, ask about them
        if entities:
            entity, entity_type = entities[0]
            if entity_type == 'PERSON':
                return f"Who is {entity}?"
            elif entity_type in ['GPE', 'LOCATION']:
                return f"Where is {entity}?"
            elif entity_type == 'ORGANIZATION':
                return f"What is {entity}?"
        
        # Check for sentences with "because", "due to", "as a result"
        if re.search(r'\bbecause\b|\bdue to\b|\bas a result\b', sentence, re.IGNORECASE):
            return f"Why {sentence.lower()}?"
        
        # Simplified approach without relying on POS tagging
        words = sentence.split()
        
        # Very simple subject extraction (first word)
        if words:
            subject = words[0]
            if subject.lower() in ['i', 'you', 'we', 'they', 'he', 'she', 'it', 'this', 'that']:
                return f"What did {subject.lower()} do?"
            else:
                return f"What about {subject}?"
        
        # Very generic fallback
        question_starters = [
            "What is described in",
            "What is mentioned about",
            "Can you explain",
            "Could you elaborate on"
        ]
        
        return f"{random.choice(question_starters)} the statement: '{sentence}'?"
    except Exception as e:
        print(f"Question generation failed: {str(e)}")
        return f"What can you tell me about: '{sentence}'?"

def paragraph_to_questions(paragraph):
    """Generate questions from a paragraph."""
    try:
        # Try the NLTK sentence tokenizer
        sentences = nltk.sent_tokenize(paragraph)
        print(f"NLTK tokenizer found {len(sentences)} sentences")
    except Exception as e:
        print(f"NLTK sentence tokenization failed: {str(e)}, using fallback")
        # Fallback to simple tokenizer if NLTK fails
        sentences = simple_sentence_tokenizer(paragraph)
        print(f"Fallback tokenizer found {len(sentences)} sentences")
    
    questions = []
    
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 4:
            continue
        
        question = generate_question_from_sentence(sentence)
        questions.append(question)
    
    return questions

# Function to format the output for Gradio
def generate_questions(paragraph):
    if not paragraph or paragraph.strip() == "":
        return "Please enter a paragraph to generate questions."
    
    print(f"Processing paragraph: {paragraph[:50]}...")
    questions = paragraph_to_questions(paragraph)
    
    if not questions:
        return "Could not generate any questions from this text. Try a longer or more detailed paragraph."
    
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

# Create Gradio interface
demo = gr.Interface(
    fn=generate_questions,
    inputs=gr.Textbox(lines=10, placeholder="Enter a paragraph to generate questions..."),
    outputs=gr.Textbox(label="Generated Questions"),
    title="Paragraph to Questions Generator",
    description="Enter a paragraph and the model will generate relevant questions based on the content.",
)

# For use as a module in other Hugging Face applications
def generate_questions_from_text(text):
    return paragraph_to_questions(text)

# Launch the app if running directly
if __name__ == "__main__":
    demo.launch()