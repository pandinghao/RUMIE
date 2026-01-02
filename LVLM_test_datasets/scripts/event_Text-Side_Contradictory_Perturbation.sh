python data_process/process_m2e2.py --text_event_json rumie_datasets/m2e2/text_noise/Text-Side_Contradictory_Perturbation/text_multimedia_event.json --out_dir data_process/processed_data/event/Text-Side_Contradictory_Perturbation
python LVLM_test_datasets/convert_m2e2.py \
  --input_file data_process/processed_data/event/Text-Side_Contradictory_Perturbation/m2e2_test_ED.json \
  --output_file LVLM_test_datasets/m2e2/Text-Side_Contradictory_Perturbation/test.json \
  --image_base_dir datasets/m2e2/m2e2_rawdata/image/image