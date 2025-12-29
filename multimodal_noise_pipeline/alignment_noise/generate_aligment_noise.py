multimodal_distractor_prompt = '''
{example data} The above is a Multimodal Event Detection data entry, consisting of a sentence, associated image, 
and annotated events.

Now, you need to augment the sentence by adding descriptive content related to background elements, 
secondary objects, or environmental details visible in the image.
The added content should:
1. Be informative but event-irrelevant.
2. Not introduce new actions, changes, or event triggers.
3. Increase contextual complexity and distraction.

Please output in the following format, without any additional content:
{"sentence": "", "events": []}
'''

visual_substitution_prompt = '''
{example data} The above is a Multimodal Event Detection data entry, where "sentence" contains the textual description,
and "events" contains the event trigger information.

Now, you need to rewrite some non-trigger expressions in the sentence so that:
1. The event trigger words remain unchanged.
2. The rewritten expressions become more abstract, ambiguous, or underspecified.
3. The meaning can still be resolved with the help of the associated image.
4. No new event information is introduced.

Please output in the following format, without any additional content:
{"sentence": "", "events": []}
'''

multimodal_decouple_prompt = '''
{example data} The above is a Multimodal Event Detection data entry, where "sentence" contains the textual description, 
"events" contains the event trigger information, and an image is implicitly associated with the sentence.

Now, you need to keep the original sentence unchanged, while adding additional textual content that is 
plausible but weakly aligned or potentially inconsistent with the visual scene.
The added content should:
1. Not introduce new event triggers or event types.
2. Not contradict the original event explicitly.
3. Increase cross-modal ambiguity.

Please output in the following format, without any additional content:
{"sentence": "", "events": []}
'''