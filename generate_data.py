import pandas as pd
import random
import os

data = {
    'happy': [
        "I am very happy today", "This is the best day of my life!", "I feel amazing",
        "Such a wonderful experience", "I am so joyful", "Fantastic news!",
        "I couldn't be happier", "Everything is going great", "I love this!",
        "This makes me smile", "I am delighted", "Feeling on top of the world",
        "It's a beautiful day", "I'm thrilled with the results", "Absolutely wonderful",
        "This is pure joy", "So glad to hear this", "Ecstatic about the new project",
        "I am extremely pleased", "What a lovely surprise"
    ],
    'sad': [
        "I feel so sad", "This is heartbreaking", "I am devastated",
        "I feel down and depressed", "Such miserable news", "I am crying right now",
        "I feel terrible", "Nothing is going right", "I am completely lost",
        "This breaks my heart", "Feeling very gloomy", "I have lost all hope",
        "I am in tears", "Why does everything hurt", "I feel so lonely",
        "This is a tragedy", "I can't stop crying", "Feeling very empty inside",
        "I am deeply sorrowful", "Life feels so hard right now"
    ],
    'angry': [
        "I am really frustrated and angry", "This makes my blood boil", "I am furious!",
        "How dare they do this", "I hate this so much", "This is completely unacceptable",
        "I am outraged", "I can't stand it anymore", "So annoyed right now",
        "This is maddening", "I am getting genuinely angry", "Infuriating situation",
        "This is the worst", "I am totally pissed off", "Don't ever do that again",
        "I am losing my temper", "What a complete idiot", "This is infuriating",
        "I am raging right now", "Completely annoyed and frustrated"
    ],
    'fear': [
        "I am so scared", "This is terrifying", "I feel extremely anxious",
        "I am afraid of what might happen", "This creeps me out", "I am panicking",
        "Feeling very nervous", "I am terrified", "This is frightening",
        "I am trembling with fear", "I feel so worried", "This gives me nightmares",
        "I am horror-struck", "Feeling spooked", "I am dreading this",
        "Cannot stop shivering in fear", "This makes me so anxious", "I am petrified",
        "Really scared for my life", "I feel a sense of impending doom"
    ],
    'surprise': [
        "Wow! I didn't expect that", "What a shock!", "I am completely amazed",
        "This is astonishing", "I am surprised", "No way! Really?",
        "I can't believe this", "Whoa, that is incredible", "I am stunned",
        "Mind blown!", "This caught me off guard", "Completely unexpected",
        "Oh my goodness, wow", "I am in awe", "That is baffling",
        "What a revelation", "I am totally shocked", "Didn't see that coming",
        "Fascinating discovery", "I am speechless!"
    ],
    'neutral': [
        "I am going to the store", "The sky is blue", "This is a book",
        "I will eat lunch now", "Here is the report", "The meeting is at 5 PM",
        "I am reading an article", "This is just a regular day", "I am drinking water",
        "The car is parked outside", "I need to do my laundry", "It is 10 AM",
        "I am writing an email", "The temperature is 20 degrees", "I walked the dog",
        "Here are the keys", "I watched a video", "It's a tuesday",
        "The train arrived on time", "She said hello"
    ]
}

# Duplicate the data a few times and add slight variations to generate a bigger dataset.
samples = []
for emotion, sentences in data.items():
    for sentence in sentences:
        samples.append([sentence, emotion])
        samples.append([sentence.lower(), emotion])
        samples.append([sentence + "!", emotion])
        if "." not in sentence and "!" not in sentence:
             samples.append([sentence + ".", emotion])

random.shuffle(samples)

df = pd.DataFrame(samples, columns=['text', 'emotion'])
os.makedirs('data', exist_ok=True)
df.to_csv('data/dataset.csv', index=False)
print(f"Dataset generated with {len(df)} samples at 'data/dataset.csv'")
