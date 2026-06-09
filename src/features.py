import re
import math
import itertools
import numpy as np
import pandas as pd

def count_syllables(word):
    word = word.lower()
    if len(word) <= 3: return 1
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e'): count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy': count += 1
    return max(1, count)

def extract_base_features(df):
    print(f"⚙️ Extracting 35+ Linguistic Features for {len(df)} rows...")
    df = df.copy()
    
    def clean(text): return str(text).strip()
    def get_words(text): return re.findall(r"\w+(?:'\w+)?", clean(text).lower())
    def get_sentences(text): return re.split(r'[.!?]+', clean(text))
    
    def readability_stats(row):
        text = clean(row['Essay'])
        words = get_words(text)
        sentences = [s for s in get_sentences(text) if len(s.split()) > 0]
        
        n_words = len(words)
        n_sentences = len(sentences)
        n_syllables = sum(count_syllables(w) for w in words)
        n_complex = sum(1 for w in words if count_syllables(w) >= 3)
        n_chars = len(text)
        
        if n_words == 0 or n_sentences == 0:
            return pd.Series([0, 0, 0, 0, 0])
        
        avg_sent_len = n_words / n_sentences
        avg_word_len = n_chars / n_words
        avg_syll_word = n_syllables / n_words
        
        flesch_ease = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syll_word)
        gunning_fog = 0.4 * (avg_sent_len + 100 * (n_complex / n_words))
        smog = 1.0430 * math.sqrt(n_complex * (30 / n_sentences)) + 3.1291 if n_sentences >= 30 else 0
        
        return pd.Series([flesch_ease, gunning_fog, smog, avg_sent_len, n_complex])

    def lexical_stats(row):
        words = get_words(row['Essay'])
        if len(words) == 0: return pd.Series([0, 0, 0])
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / len(words)
        return pd.Series([len(unique_words), ttr, long_word_ratio])

    def error_proxies(row):
        text = clean(row['Essay'])
        double_spaces = len(re.findall(r'  ', text))
        space_punct = len(re.findall(r' \.', text)) + len(re.findall(r' ,', text))
        sentences = [s.strip() for s in get_sentences(text) if s.strip()]
        lower_starts = sum(1 for s in sentences if len(s) > 0 and s[0].islower())
        repeated = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
        return pd.Series([double_spaces, space_punct, lower_starts, repeated])

    def prompt_overlap(row):
        p_words = set(get_words(row.get('Prompt', '')))
        e_words = set(get_words(row.get('Essay', '')))
        stopwords = {'the','and','is','in','at','of','a','an','to','for','with','on'}
        p_words -= stopwords
        overlap = len(p_words.intersection(e_words))
        ratio = overlap / len(p_words) if len(p_words) > 0 else 0
        return pd.Series([overlap, ratio])

    df['char_count'] = df['Essay'].apply(lambda x: len(clean(x)))
    df['word_count'] = df['Essay'].apply(lambda x: len(get_words(x)))
    df['sentence_count'] = df['Essay'].apply(lambda x: len([s for s in get_sentences(x) if len(s.split())>0]))
    df['paragraph_count'] = df['Essay'].apply(lambda x: len(re.findall(r'\n+', clean(x))) + 1)
    
    df['question_marks'] = df['Essay'].apply(lambda x: str(x).count('?'))
    df['exclamation_marks'] = df['Essay'].apply(lambda x: str(x).count('!'))
    df['semicolons'] = df['Essay'].apply(lambda x: str(x).count(';'))
    df['quotes'] = df['Essay'].apply(lambda x: str(x).count('"'))
    
    df[['flesch_ease', 'gunning_fog', 'smog', 'avg_sent_len', 'n_complex']] = df.apply(readability_stats, axis=1)
    df[['unique_words', 'ttr', 'long_word_ratio']] = df.apply(lexical_stats, axis=1)
    df[['err_double_space', 'err_space_punct', 'err_lower_start', 'err_repeated']] = df.apply(error_proxies, axis=1)
    df[['prompt_overlap_count', 'prompt_overlap_ratio']] = df.apply(prompt_overlap, axis=1)
    
    return df

def add_advanced_features(df_train, df_test):
    print("🚀 Generating Advanced Features (Squared, Interactions, Ratios)...")
    golden_features = [
        'deberta_pred', 'word_count', 'sentence_count', 'paragraph_count', 
        'avg_sent_len', 'unique_words', 'n_complex', 'ttr', 
        'flesch_ease', 'gunning_fog', 'smog'
    ]
    golden_features = [c for c in golden_features if c in df_train.columns]

    def engineer(df):
        df = df.copy()
        for col in golden_features:
            df[f'{col}_sq'] = df[col] ** 2
            
        for col1, col2 in itertools.combinations(golden_features, 2):
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            epsilon = 1e-6
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + epsilon)
            df[f'{col2}_div_{col1}'] = df[col2] / (df[col1] + epsilon)
        return df

    df_train = engineer(df_train)
    df_test = engineer(df_test)
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test = df_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_train, df_test
