# MeLiSA: Mercado Libre for Sentiment Analysis

**  **NOTE: THIS FILE IS UNDER CONSTRUCTION** **

This is the main repository of the MeLiSA dataset, which is designed to perform research in Latin American Spanish and Portuguese text classification. The dataset can be accessed with the [🤗 Datasets library](https://huggingface.co/docs/datasets/) with a few lines of Python code:
```Python
from datasets import load_dataset

dataset = load_dataset("lpsc-fiuba/melisa", "all_languages")
# you can use any of the following config names as a second argument: 
# "all_languages", "es", "pt"
```
For more details on how to access to the data visit our [Huggingface organization page](https://huggingface.co/lpsc-fiuba). This page also contains finetunned models on this dataset.

The [code](./code) used to download the data is also provided in this repository.

- **Point of Contact:** lestienne@fi.uba.ar

## Dataset Description

We provide a Mercado Libre product reviews dataset for spanish and portuguese text classification. The dataset contains reviews in these two languages collected between August 2020 and January 2021. Each record in the dataset contains the review content and title, the star rating, the country where it was pubilshed and the product category (arts, technology, etc.). The corpus is roughly balanced across stars, so each star rating constitutes approximately 20% of the reviews in each language.

|   || Spanish ||| Portugese ||
|---|:------:|:----------:|:-----:|:------:|:----------:|:-----:|
|   | Train  | Validation | Test  | Train  | Validation | Test  |
| 1 | 88.425 | 4.052      | 5.000 | 50.801 | 4.052      | 5.000 |
| 2 | 88.397 | 4.052      | 5.000 | 50.782 | 4.052      | 5.000 |
| 3 | 88.435 | 4.052      | 5.000 | 50.797 | 4.052      | 5.000 |
| 4 | 88.449 | 4.052      | 5.000 | 50.794 | 4.052      | 5.000 |
| 5 | 88.402 | 4.052      | 5.000 | 50.781 | 4.052      | 5.000 |

Table shows the number of samples per star rate in each split. There is a total of 442.108 training samples in spanish and 253.955 in portuguese. We limited the number of reviews per product to 30 and we perform a ranked inclusion of the downloaded reviews to include those with rich semantic content. In these ranking, the lenght of the review content and the valorization (difference between likes and dislikes) was prioritized. For more details on this process, see (CITATION).

Reviews in spanish were obtained from 8 different Latin Amercian countries (Argentina, Colombia, Peru, Uruguay, Chile, Venezuela and Mexico), and portuguese reviews were extracted from Brasil. To match the language with its respective country, we applied a language detection algorithm based on the works of Joulin et al. (2016a and 2016b) to determine the language of the review text and we removed reviews that were not written in the expected language.

### Data Fields

- `country`: The string identifier of the country. It could be one of the following: `MLA` (Argentina), `MCO` (Colombia), `MPE` (Peru), `MLU` (Uruguay), `MLC` (Chile), `MLV` (Venezuela), `MLM` (Mexico) or `MLB` (Brasil).
- `category`: String representation of the product's category. It could be one of the following:
    - Hogar / Casa
    - Tecnologı́a y electrónica / Tecnologia e electronica
    - Salud, ropa y cuidado personal / Saúde, roupas e cuidado pessoal
    - Arte y entretenimiento / Arte e Entretenimiento
    - Alimentos y Bebidas / Alimentos e Bebidas
- `review_content`: The text content of the review.
- `review_title`: The text title of the review.
- `review_rate`: An int between 1-5 indicating the number of stars.

### Data Splits

Each language configuration comes with it's own `train`, `validation`, and `test` splits. The `all_languages` split is simply a concatenation of the corresponding split across all languages. That is, the `train` split for `all_languages` is a concatenation of the `train` splits for each of the languages and likewise for `validation` and `test`.

### Personal and Sensitive Information
Mercado Libre Reviews are submitted by users with the knowledge and attention of being public. The reviewer ID's included in this dataset are anonymized, meaning that they are disassociated from the original user profiles. However, these fields would likely be easy to deannoymize given the public and identifying nature of free-form text responses.

### Discussion of Biases
The data included here are from unverified consumers. Some percentage of these reviews may be fake or contain misleading or offensive language. 

### Other Known Limitations
The dataset is constructed so that the distribution of star ratings is roughly balanced. This feature has some advantages for purposes of classification, but some types of language may be over or underrepresented relative to the original distribution of reviews to acheive this balance.
[More Information Needed]




## Additional Information

Published by Lautaro Estienne, Matías Vera and Leonardo Rey Vega. Managed by the Signal Processing in Comunications Laboratory of the Electronic Department at the Engeneering School of the Buenos Aires University (UBA).

### Licensing Information
<!-- Amazon has licensed this dataset under its own agreement, to be found at the dataset webpage here:
https://docs.opendata.aws/amazon-reviews-ml/license.txt -->
[More Information Needed]

### Citation Information
Please cite the following paper if you found this dataset useful:

(CITATION)
[More Information Needed]
