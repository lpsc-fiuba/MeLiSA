#! /bin/bash

mkdir products parts parts-2 peru_parts ven_parts
python 01-get_products_list.py
python 02-remove_duplicates_and_get_reviews.py --partsdir "parts" --countries "MLA" "MCO" "MPE" "MLU" "MLC" "MLM" "MLV" "MLB"
python 02-remove_duplicates_and_get_reviews.py --partsdir "parts-2" --countries "MLA" "MCO" "MPE" "MLU" "MLC" "MLM" "MLV" "MLB"
python 02-remove_duplicates_and_get_reviews.py --partsdir "peru_parts" --countries "MPE"
python 02-remove_duplicates_and_get_reviews.py --partsdir "ven_parts" --countries "MLV"
python 03-get_parts_in_csv.py
python 04-process_csv.py
python 06-train-dev-test-split.py