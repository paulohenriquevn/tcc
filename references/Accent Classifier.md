Speech Portuguese (Brazilian) Accent Classifier
🎙️🤖🇧🇷

This project is a speech accent classifier that distinguishes between Portuguese (Brazilian) and other accents.

Project Overview
This application uses a trained model to classify speech accents into two categories:

Portuguese (Brazilian)
Other
The model is based on the author's work [results] and utilizes the Portuguese portion of the Common Voice dataset (version 11.0) from Mozilla Foundation.

Dataset
The project uses the Portuguese subset of the Common Voice dataset:

Dataset: "mozilla-foundation/common_voice_11_0", "pt"
Brazilian accents included in the dataset:

Português do Brasil, Região Sul do Brasil
Paulistano
Paulista, Brasileiro
Carioca
Mato Grosso
Mineiro
Interior Paulista
Gaúcho
Nordestino
And various regional mixes
Model and Processor
The project utilizes the following model and processor:

Base Model: "facebook/wav2vec2-base-960h"
Processor: Wav2Vec2Processor.from_pretrained
Model Versions
Was trained three versions of the model with different configurations:

(OLD) v 1.1:

Epochs: 3
Training samples: 1000
Validation samples: 200
(OLD) v 1.2:

Epochs: 10
Training samples: 1000
Validation samples: 500
(NEW) v 1.3:

Epochs: 20
Training samples: 5000
Validation samples: 1000
All models were trained using high RAM GPU on Google Colab Pro.

Model Structure (files)
Each version of the model includes the following files: results config.json | preprocessor_config.json | model.safetensors | special_tokens_map.json | tokenizer_config.json | vocab.json

How to Use
Test with recording or uploading an audio file. To test, I recommend short sentences.

License
This project is licensed under the Eclipse Public License 2.0 (ECL-2.0).

Developer Information
Developed by Ramon Mayor Martins (2024)

Email: rmayormartins@gmail.com
Homepage: https://rmayormartins.github.io/
Twitter: @rmayormartins
GitHub: https://github.com/rmayormartins
Acknowledgements
Special thanks to Instituto Federal de Santa Catarina (Federal Institute of Santa Catarina) IFSC-São José-Brazil.

Contact
For any queries or suggestions, please contact the developer using the information provided above.