<h1 align="center"> Natural Language Processing  with Hugging Face Transformers by Andhika Laksmana </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Andhika Laksmana Putra Alka

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I hate doing my work, and i want to be free better be with my girlfriend")
```

Result : 

```
[{'label': 'NEGATIVE', 'score': 0.9983258843421936}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Attack on Titan is a popular anime series known for its intense storyline, emotional depth, and stunning animation.",
    candidate_labels=["Korean culture", "Games", "entertainment"],
)
```

Result : 

```
{'sequence': 'Attack on Titan is a popular anime series known for its intense storyline, emotional depth, and stunning animation.',
 'labels': ['entertainment', 'Games', 'Korean culture'],
 'scores': [0.9852999448776245, 0.008583945222198963, 0.006116130854934454]}
```

Analysis on example 2 : 

The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
# TODO :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "If you watch anime you will",
    max_length=30, # you can change this
    num_return_sequences=2, # and this too
)
```

Result : 

```
[{'generated_text': "If you watch anime you will notice that the following image doesn't really show up so just be sure to not miss anything, just keep in mind that"},
 {'generated_text': 'If you watch anime you will see.\n\nThe reason I am choosing not to watch anime is because it will be more interesting to watch and more'}]
```

Analysis on example 3 : 

The text generation model produces coherent and imaginative continuations of a cooking-themed prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Watching anime makes me feel so <mask> inside.", top_k=4)
```

Result : 

```
[{'score': 0.07649342715740204,
  'token': 1372,
  'token_str': ' happy',
  'sequence': 'Watching anime makes me feel so happy inside.'},
 {'score': 0.053488124161958694,
  'token': 1522,
  'token_str': ' safe',
  'sequence': 'Watching anime makes me feel so safe inside.'},
 {'score': 0.051387492567300797,
  'token': 5074,
  'token_str': ' sad',
  'sequence': 'Watching anime makes me feel so sad inside.'},
 {'score': 0.048251621425151825,
  'token': 4736,
  'token_str': ' sick',
  'sequence': 'Watching anime makes me feel so sick inside.'}]
```

Analysis on example 3.5 : 

The fill-mask pipeline accurately infers masked words based on context. The top result "stole" makes sense, supported by a high confidence score. Other predictions are also contextually appropriate, illustrating the model's nuanced understanding of sentence structure and intent.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Andhika and I become a mentee in Infinite Learning in Batam.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.99609256),
  'word': 'Andhika',
  'start': 11,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.98432314),
  'word': 'Infinite Learning',
  'start': 44,
  'end': 61},
 {'entity_group': 'LOC',
  'score': np.float32(0.983708),
  'word': 'Batam',
  'start': 65,
  'end': 70}]
```

Analysis on example 4 : 

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Who is the Anemo Archon in Genshin Impact?"
context = "In Genshin Impact, the Anemo Archon is Barbatos, who is also known as Venti, the free-spirited bard of Mondstadt."
qa_model(question=question, context=context)
```

Result : 

```
{'score': 0.9889916181564331, 'start': 39, 'end': 47, 'answer': 'Barbatos'}
```

Analysis on example 5 : 

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Genshin Impact is an open-world action role-playing game developed by miHoYo. It features a fantasy-based environment and an action-based battle system using elemental magic and character-switching. The game is free-to-play and monetized through gacha game mechanics. It is set in the world of Teyvat, which is home to seven distinct nations, each of which is tied to a different element and ruled by a different god. The story follows the Traveler, who has traveled across many worlds with their twin sibling before becoming separated in Teyvat. The player assumes the role of the Traveler and journeys across the nations of Teyvat in search of their lost sibling, accompanied by a floating companion named Paimon. Along the way, the Traveler becomes involved in the affairs of Teyvat's nations and gods. The game is praised for its beautiful world design, engaging combat, and rich storytelling.
    """
)
```

Result : 

```
[{'summary_text': ' Genshin Impact is an open-world action role-playing game developed by miHoYo . It features a fantasy-based environment and an action-based battle system using elemental magic and character-switching . The game is set in the world of Teyvat, home to seven distinct nations, each of which is tied to a different element and ruled by a different god .'}]

```

Analysis on example 6 :

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Menurutmu aku itu penyuka anime atau bukan?")
```

Result : 

```
[{'translation_text': 'Tu penses que je suis une animeuse ou pas ?'}]

```

Analysis on example 7 :

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.