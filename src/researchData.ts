export const attentionPaperChapters = [
  {
    label: 'The problem',
    title: 'The Hallway Telephone Game (Why RNNs Failed)',
    body: 'Imagine 50 people standing in a single-file line down a long hallway. Person 1 whispers a secret message to Person 2, who whispers it to Person 3, all the way to Person 50. This was how AI read text before 2017 using Recurrent Neural Networks (RNNs). Two huge problems crippled this system: First, by the time the message reached Person 50, details from the beginning were garbled and forgotten. Second, Person 50 could not start until everyone ahead of them finished. Even though modern graphics cards (GPUs) have over 10,000 parallel computing cores, an RNN forced 9,999 of them to sit idle, waiting in single file.',
    analogy: 'A 50-person game of telephone down a narrow corridor: slow, prone to memory loss, and wasting 99% of computer horsepower.',
    takeaway: 'Reading one word at a time forced supercomputers to wait in a single-file line and made long sentences blur into fog.'
  },
  {
    label: 'The big idea',
    title: 'Attention Is All You Need (Knocking Down the Walls)',
    body: 'In 2017, a team at Google had a radical idea: what if we knock down the hallway walls and put all 50 people in one open circular room? Instead of whispering in a chain, give everyone laser pointers. In a single instant, every word can shine its laser directly at whichever other words hold relevant clues! Because every connection happens simultaneously, all 10,000 GPU cores can fire in parallel. Training that previously took months finished in just 3.5 days. The authors titled their paper with bold simplicity: "Attention Is ALL You Need"—recurrence was completely discarded.',
    analogy: 'Replacing a whispering telephone line with 50 laser pointers in an open amphitheater: instant, direct, and fully parallel.',
    takeaway: 'Attention completely replaced sequential chains with instant, parallel spotlights connecting any two words.'
  },
  {
    label: 'The mechanism',
    title: 'Questions, Badges & Notes (The Q, K, V Dialogue)',
    body: 'How do words figure out whose laser pointer to follow? Every word gets three simple items: (1) A Query ("What question am I asking?"), (2) A Key ("What is on my nametag to match questions?"), and (3) A Value ("What message do I share if there is a match?"). For example, when the pronoun "it" asks: "What animal was mentioned?", the noun "cat" waves its nametag: "I am a feline animal!". Their match score is high, so "it" borrows the rich meaning of "cat". By repeating this for every word, sentences become rich, self-explaining maps of meaning.',
    analogy: 'A lively research mixer: Query is your question, Key is someone’s nametag, and Value is the notebook page they hand you.',
    takeaway: 'Query asks, Key matches, and Value delivers the knowledge.'
  },
  {
    label: 'The volume knob',
    title: 'The Volume Control (Why the Authors Divided by √dₖ)',
    body: 'When Query and Key vectors compare notes across 64 or 128 dimensions, adding up all those numbers creates huge scores. If you feed huge numbers into a softmax decision, it acts like a microphone turned up too loud: one word screams at 99.9% volume, deafening the computer to every other word! When the model is deaf, learning freezes (the vanishing gradient trap). The authors added a simple, brilliant master volume knob: divide the raw score by √dₖ (the square root of the dimension size). This brings the volume back down to a pleasant conversation so all relevant clues are heard in harmony.',
    analogy: 'Turning down the master gain on an audio mixer before loud shouting distorts the speakers and blows out your ears.',
    takeaway: 'Dividing by √dₖ is a master volume dial: it prevents one word from screaming and deafening the learning process.'
  },
  {
    label: 'Many views',
    title: 'The 8 Pairs of Glasses (Multi-Head Attention)',
    body: 'A human reader doesn’t look at a sentence for just one thing. We track grammar, pronoun references, emotion, and facts all at once. If an AI only had one attention head, it would have to average all those different relationships together into a muddy blur. The authors gave the Transformer 8 distinct "pairs of glasses" (heads) running concurrently. One pair tracks which verb goes with which noun, another links pronouns to names, and another watches punctuation. Combining these 8 distinct viewpoints gives the model deep, nuanced comprehension.',
    analogy: 'Giving an investigator 8 specialized lenses: one for grammar, one for pronouns, one for emotion, and one for facts.',
    takeaway: 'Multiple heads give the model several specialized pairs of eyes so different relationships don’t get blurred together.'
  },
  {
    label: 'Order matters',
    title: 'The Rhythm Clocks (Positional Encodings Made Simple)',
    body: 'Because laser pointers shine everywhere at once, attention has no natural concept of time or sequence. If you shuffled the words like a deck of playing cards, attention would see the exact same pile! But "dog bites man" is very different from "man bites dog". To fix this without adding heavy learned tables, the authors stamped each position with musical rhythm clocks: fast ticks (high frequency) for nearby neighbor words, and slow chimes (low frequency) for paragraph-level position. By listening to the clock harmony, the model immediately knows how many words separate any two points.',
    analogy: 'Stamping every seat with an orchestra clock: high flutes tell you who sits next to you, deep tubas tell you what section you are in.',
    takeaway: 'Smooth musical frequencies give each seat a timestamp so the model knows who came first, second, and third.'
  },
  {
    label: 'The translation room',
    title: 'The Bilingual Newsroom (Encoder meets Decoder)',
    body: 'The 2017 Transformer was designed for translating English to German. It split the work between two specialized teams: The English Reading Desk (The Encoder) reads the entire English sentence in one parallel glance and creates a rich bulletin board of facts. Then, the German Writing Desk (The Decoder) writes the German translation one word at a time. On every new word, the German writer points a microphone at the English bulletin board (Cross-Attention) to pull the exact English meaning needed for that specific word.',
    analogy: 'An English research panel preparing briefing notes, while a German journalist types the live report using those notes.',
    takeaway: 'In Cross-Attention, the German writer asks questions and the English panel supplies the answers.'
  },
  {
    label: 'The 2024 catch',
    title: 'The Modern Catch (Why the 2017 Win Created the KV Cache Crisis)',
    body: 'During training, attention is wonderfully parallel because the entire document is already written. But when a modern chatbot talks to you live, it must generate text the old-fashioned way: one word at a time. To keep from re-reading the entire conversation from scratch for every single word, AI servers must store every past Key and Value note on a massive memory shelf: The KV Cache. Today, serving millions of users with 100,000-word context windows is threatening to run out of GPU memory. The very mechanism that made 2017 training super-fast created the primary memory engineering bottleneck of 2024!',
    analogy: 'Taking notes during an interview is fast, but keeping 100,000 notebooks stacked on your desk for every caller burns through all your office space.',
    takeaway: 'The laser pointer made training instantaneous, but storing past targets during live chat created the modern KV Cache crisis.'
  }
];

export const upcomingPapers = [
  ['FlashAttention (Dao et al. 2022)', 'How IO-aware exact attention keeps data in high-speed GPU SRAM, eliminating memory bottlenecks.'],
  ['Mamba & S4 (Gu & Dao 2023)', 'How selective state-space models replace the heavy KV cache shelf with a compact, rolling summary notebook.'],
  ['Chinchilla (Hoffmann et al. 2022)', 'The golden ratio of model size to training data: why early models were starved of tokens.'],
  ['PagedAttention & vLLM (Kwon et al. 2023)', 'Applying operating system virtual memory paging to stop KV cache memory waste and fragmentation.']
];

export const attentionPaperEli5 = [
  ['The problem', 'Before this paper, computers read books like a slow game of telephone: person 1 whispered to person 2, who whispered to person 3. It was slow, and by person 50, everyone forgot how the story started.'],
  ['The big idea', 'Instead of whispering in a line, give all 50 words laser pointers! In one instant, every word shines its pointer directly at whoever has the best clues.'],
  ['The mechanism', 'Every word gets a Question (Query), a Nametag (Key), and a Message (Value). Questions match to nametags, and messages get shared.'],
  ['The volume knob', 'If everyone shouts at once, the room is too loud and you go deaf. The authors added a volume knob (divide by √d) so words talk at a normal speaking volume.'],
  ['Many views', 'Give the computer 8 pairs of glasses. One pair spots grammar, another spots names, and another watches the mood of the sentence.'],
  ['Order matters', 'If you dump word cards on a table, you need little seat numbers. Musical ticking clocks tell the computer which word came first, second, and third.'],
  ['The translation room', 'An English team reads the whole English sentence at once. Then a German writer types the German words one by one, asking the English team for clues when needed.'],
  ['The 2024 catch', 'Writing one word at a time means saving lots of past clue cards in computer memory. That is why modern AI systems need giant memory cards today!']
];

export const telephoneVsLaserData = {
  rnn: {
    title: 'The Telephone Game (RNN / LSTM)',
    era: '1986 – 2016 Paradigm',
    analogy: 'Imagine 50 people standing in a line. Person 1 whispers a secret to Person 2, who whispers to Person 3, all the way to Person 50.',
    points: [
      {
        kicker: 'BOTTLENECK 01',
        title: 'The Single-File Line (Hardware Starvation)',
        desc: 'Word 50 cannot start until Word 49 finishes whispering. Modern GPUs have over 10,000 computing cores, but an RNN leaves 9,999 of them sitting empty. Training took weeks or months.'
      },
      {
        kicker: 'BOTTLENECK 02',
        title: 'Memory Fog (Information Decay)',
        desc: 'By the 50th whisper, nuances from the opening of the sentence are squashed and lost. Distant context fades away like an echo.'
      }
    ],
    summaryBadge: 'Single-file: 1 word at a time · High memory decay'
  },
  transformer: {
    title: 'The Laser Pointer (Transformer Attention)',
    era: '2017 Breakthrough (Attention Is All You Need)',
    analogy: 'Give all 50 people laser pointers in an open room. In a single instant, every person points directly at whoever holds relevant clues.',
    points: [
      {
        kicker: 'BREAKTHROUGH 01',
        title: 'Full Hardware Parallelism',
        desc: 'All 50 words find their relationships at the exact same moment via parallel matrix math. Every GPU core fires at once, slashing training time by 10x to 100x.'
      },
      {
        kicker: 'BREAKTHROUGH 02',
        title: 'Zero Memory Fog (Direct Lookups)',
        desc: 'Word 50 connects to Word 1 with the exact same crystal-clear precision as Word 49. No intermediate whispering to distort the facts.'
      },
      {
        kicker: 'THE 2024 INFERENCE CATCH',
        title: 'The KV Cache Bottleneck',
        desc: 'Training is parallel because the full book is already written. But during live chat with a user, text must be generated one word at a time. Storing past laser targets creates the massive KV Cache memory pressure in Module 2!'
      }
    ],
    summaryBadge: 'Parallel: All words at once · Crystal-clear direct connections'
  }
};

export const volumeDialDemo = {
  title: 'The Microphone Volume Dial (Why Scaling √dₖ is Required)',
  analogyStory: 'Imagine a choir of 64 or 128 singers. If everyone shouts their score directly into the microphone at full blast, the audio console redlines and distorts. In neural networks, when scores get too large, Softmax acts like an ear-splitting screamer: ONE word claims 99.9% of the volume, and all other words drop to 0%. The network goes completely deaf to other clues, and gradients vanish! The authors turned down the master volume knob (dividing by √dₖ) so all clues can be heard in balanced harmony.',
  sentence: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired'],
  queryWord: 'it',
  scenarios: {
    scaled: {
      label: 'Balanced Volume Dial (Scaling ON: divide by √dₖ)',
      status: '🟢 Harmonious Chorus (Healthy Learning)',
      explanation: 'With the volume knob engaged, scores remain at comfortable speaking levels. The pronoun "it" distributes focus intelligently: 58% on "cat" (the true subject), 22% on "tired" (its current state), and 12% on "sat" (its action). The computer learns smoothly!',
      weights: [
        { word: 'cat', pct: 58, role: 'Main Referent (Who is it?)', isMain: true },
        { word: 'tired', pct: 22, role: 'Descriptive State (How is it?)', isMain: false },
        { word: 'sat', pct: 12, role: 'Associated Action', isMain: false },
        { word: 'mat', pct: 5, role: 'Nearby Context', isMain: false },
        { word: 'The', pct: 3, role: 'Grammar Article', isMain: false }
      ],
      gradientHealth: 'HEALTHY GRADIENTS: Learning signals flow freely back to update the brain.'
    },
    unscaled: {
      label: 'Volume Too Loud (Scaling OFF: Raw Unscaled Shouting)',
      status: '🔴 Screaming Contest Distortion (Vanishing Gradient)',
      explanation: 'Without the volume knob, the raw numbers blow out the soundboard. The highest score balloons to +32.0. Softmax over-amplifies this single winner, giving "cat" 99.98% volume and silencing everything else. The AI goes completely deaf to "tired" and "sat", backprop gradient drops to 0.00001, and learning freezes!',
      weights: [
        { word: 'cat', pct: 99.98, role: 'Loud Screamer (Drowning everything out)', isMain: true },
        { word: 'tired', pct: 0.01, role: 'Silenced (Deafened)', isMain: false },
        { word: 'sat', pct: 0.01, role: 'Silenced (Deafened)', isMain: false },
        { word: 'mat', pct: 0.00, role: 'Silenced (Deafened)', isMain: false },
        { word: 'The', pct: 0.00, role: 'Silenced (Deafened)', isMain: false }
      ],
      gradientHealth: '⚠️ VANISHING GRADIENT COLLAPSE: Softmax is saturated! Derivative ≈ 0. Training stalls.'
    }
  }
};

export interface EightGlassesItem {
  id: string;
  headNumber: number;
  name: string;
  badge: string;
  color: string;
  question: string;
  focusedWords: string[];
  explanation: string;
  analogy: string;
}

export const eightGlassesData: EightGlassesItem[] = [
  {
    id: 'glasses-1',
    headNumber: 1,
    name: 'The Pronoun Detective',
    badge: 'Coreference Resolution',
    color: '#9a89ff',
    question: 'When a pronoun appears, who does it actually point to?',
    focusedWords: ['cat', 'it'],
    explanation: 'Finds that the word "it" at position 8 refers directly to "cat" at position 2. Without this lens, the model would lose track of who is acting in the second half of the sentence.',
    analogy: 'A private detective shining a magnifying glass from a pronoun back to the person or animal who owns it.'
  },
  {
    id: 'glasses-2',
    headNumber: 2,
    name: 'The Action Linker',
    badge: 'Subject-Verb Agreement',
    color: '#6ed2bd',
    question: 'Who performed this action, and what did they do?',
    focusedWords: ['cat', 'sat'],
    explanation: 'Pairs the subject "cat" directly with the verb "sat". In complex sentences with lots of filler clauses, this lens keeps the core action tethered to the main character.',
    analogy: 'An anchor rope tying the character to their action so they don’t drift apart in long paragraphs.'
  },
  {
    id: 'glasses-3',
    headNumber: 3,
    name: 'The Cause & Effect Inspector',
    badge: 'Causal Relationships',
    color: '#d8f26e',
    question: 'Why did this happen? What was the reason?',
    focusedWords: ['sat', 'because', 'tired'],
    explanation: 'Connects the physical action "sat" through the pivot word "because" to the underlying reason: "tired". The model understands motivation and narrative cause.',
    analogy: 'A courtroom investigator connecting a suspect’s motive ("tired") to what actually happened ("sat").'
  },
  {
    id: 'glasses-4',
    headNumber: 4,
    name: 'The Location Scout',
    badge: 'Prepositional & Spatial Grounds',
    color: '#ff9770',
    question: 'Where did this action take place?',
    focusedWords: ['sat', 'on', 'mat'],
    explanation: 'Pins down the spatial environment: "sat" + "on" + "the mat". It creates a mental picture of the physical scene.',
    analogy: 'A stage director placing props and scenery around the actor on stage.'
  },
  {
    id: 'glasses-5',
    headNumber: 5,
    name: 'The Next-Door Neighbor Lens',
    badge: 'Local Bigrams & Syntax',
    color: '#6ed2bd',
    question: 'What immediate words sit directly to my left and right?',
    focusedWords: ['The', 'cat'],
    explanation: 'Checks immediate word pairs like "The cat" and "the mat". Verifies correct grammar articles, compound phrases, and local spelling harmony.',
    analogy: 'Peeking over the backyard fence to say hello to the person living right next door.'
  },
  {
    id: 'glasses-6',
    headNumber: 6,
    name: 'The State of Being Meter',
    badge: 'Condition & Emotion',
    color: '#9a89ff',
    question: 'What physical or emotional state is being experienced?',
    focusedWords: ['was', 'tired'],
    explanation: 'Connects the auxiliary verb "was" to the condition adjective "tired". Identifies mood, fatigue, tone, or sentiment.',
    analogy: 'A mood ring tracking whether the subject is happy, angry, or exhausted.'
  },
  {
    id: 'glasses-7',
    headNumber: 7,
    name: 'The Meaning Disambiguator',
    badge: 'Polysemy / Word Sense',
    color: '#d8f26e',
    question: 'Is "cat" a pet animal, or Caterpillar construction bulldozer?',
    focusedWords: ['cat', 'tired'],
    explanation: 'Words have multiple meanings. Bulldozers don’t get "tired", but living pets do! This lens resolves word ambiguities using surrounding clues.',
    analogy: 'A context dictionary that flips to page 47 to confirm "cat" is a furry pet, not heavy mining machinery.'
  },
  {
    id: 'glasses-8',
    headNumber: 8,
    name: 'The Global Story Conductor',
    badge: 'Document Flow & Boundaries',
    color: '#ff9770',
    question: 'Where are the sentence boundaries, and where does the thought end?',
    focusedWords: ['The', 'tired'],
    explanation: 'Looks from the very first word "The" to the final word "tired", tracking overall sentence pacing and narrative closure.',
    analogy: 'An orchestra conductor holding up the baton from the opening note to the final resolution.'
  }
];

export const rhythmClockData = {
  title: 'The Orchestra Clocks (Positional Encodings Made Simple)',
  story: 'Attention treats a sentence like Scrabble tiles jumbled in a bag: it sees "cat", "dog", "bites", "man", but doesn’t know who bit whom! Instead of adding heavy learned coordinate maps, the authors stamped every single position with a combination of musical rhythm clocks.',
  clockTypes: [
    {
      name: 'The Fast Ticking Flute (High Frequency)',
      icon: '⏱️',
      color: '#d8f26e',
      tempo: 'Ticks once per word',
      role: 'Immediate Neighbors',
      desc: 'The hand spins around with every single word. Word 2 is pointing North, Word 3 is pointing South. By checking this clock, words immediately know if they are sitting side-by-side!'
    },
    {
      name: 'The Medium Rhythm Guitar (Medium Frequency)',
      icon: '⏳',
      color: '#6ed2bd',
      tempo: 'Rolls over every 8–16 words',
      role: 'Clause Boundaries',
      desc: 'Moves at a moderate walking pace. It tells words whether they belong to the first half of the sentence ("The cat sat") or the second half ("because it was tired").'
    },
    {
      name: 'The Deep Church Bell Chime (Low Frequency)',
      icon: '🔔',
      color: '#ff9770',
      tempo: 'Drifts slowly over 100+ words',
      role: 'Paragraph & Chapter Location',
      desc: 'Only nudges a tiny fraction of an inch across long passages. It gives the model a broad, stable sense of whether it is near the opening intro or the final conclusion.'
    }
  ],
  magicInsight: {
    heading: 'The Magic of Relative Music',
    body: 'Why did the authors use sine and cosine waves? Because of high-school trigonometry! If Word 2 has clock angle α, and Word 5 has clock angle α + β, the difference between them is simply the angle β! The model does not need to memorize absolute seat numbers: it simply listens to the harmonious angle between any two clock hands to instantly know their exact distance.'
  }
};

export const translationRoomEvolution = [
  {
    id: 'room-1',
    era: '2017: The Original Transformer',
    role: 'The Two-Desk Newsroom (Encoder-Decoder)',
    analogy: 'An English team reading source documents at one table, passing notes across the room to a German reporter typing the translation at another table.',
    howItWorks: 'The Encoder reads the whole English sentence in parallel. The Decoder generates German words one at a time, using Cross-Attention to query the English notes.',
    pros: 'Brilliant for language-to-language translation and audio-to-text.',
    cons: 'Requires managing two separate models and complex cross-attention wiring.',
    models: 'Original Transformer, T5, MarianMT, Whisper',
    kvStatus: 'Dual cache: Decoder keeps its own past words, while holding a static copy of the English notes.'
  },
  {
    id: 'room-2',
    era: '2018: The Search Engine Revolution',
    role: 'The English-Only Reading Desk (Encoder-Only / BERT)',
    analogy: 'We removed the German writer completely! Just a super-smart English research team analyzing existing books and ranking search results.',
    howItWorks: 'Every word looks forward and backward simultaneously with no blindfolds. Cannot generate new stories, but understands text deeply.',
    pros: 'Undisputed champion for search ranking, embeddings, sentiment analysis, and classification.',
    cons: 'Cannot chat or generate text word-by-word.',
    models: 'BERT, RoBERTa, DeBERTa, Google Search Ranking',
    kvStatus: 'ZERO KV Cache needed! The whole sentence is processed in one quick shot without remembering past states.'
  },
  {
    id: 'room-3',
    era: '2020 – 2026: The Generative AI Boom',
    role: 'The Solo Master Writer (Decoder-Only / GPT & Claude)',
    analogy: 'We fired the English research desk! Turns out, if you just give the solo writer a typewriter and train them to guess the next word over the whole internet, they can do translation, coding, math, and chatting all by themselves!',
    howItWorks: 'Strictly autoregressive: reads the prompt, then writes one new word at a time, predicting what comes next.',
    pros: 'One unified architecture for everything: code, chat, reasoning, translation, and agentic workflows.',
    cons: 'Massive memory appetite during live serving.',
    models: 'GPT-4, Claude 3.5, LLaMA 3, Mistral, DeepSeek, Gemini',
    kvStatus: 'CRITICAL BOTTLENECK: The solo writer must save every single past word in a giant memory cabinet (The KV Cache). When conversations grow long, this cabinet explodes GPU VRAM—which is why you need Module 02!'
  }
];
