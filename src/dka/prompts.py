# Adapted from Microsoft GraphRAG (MIT License)
# Changes: knowledge-domain entity types, bilingual (Chinese + English) support

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document and a list of entity types, identify all entities of those types and all relationships among them.
Always write entity names and descriptions in the same language as the input text.

-Steps-
1. Identify all entities. For each entity extract:
   - entity_name: Name of the entity (capitalize English names)
   - entity_type: One of [{entity_types}]
   - entity_description: Concise description of the entity's role and attributes, written in the same language as the input text

   Format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From step 1, identify all clearly related entity pairs. For each pair extract:
   - source_entity: name from step 1
   - target_entity: name from step 1
   - relationship_description: why they are related, written in the same language as the input text
   - relationship_strength: integer 1-10 indicating strength

   Format: ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Output all entities and relationships as a single list delimited by **##**.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: CONCEPT,MODEL,METHOD
Text:
BERT is a pre-trained language model based on the Transformer architecture. It uses bidirectional self-attention to encode context from both directions. Fine-tuning BERT on downstream tasks achieves state-of-the-art results on benchmarks like SQuAD.
######################
Output:
("entity"<|>BERT<|>MODEL<|>Bidirectional pre-trained language model based on Transformer, fine-tuned for downstream NLP tasks)
##
("entity"<|>TRANSFORMER<|>MODEL<|>Neural network architecture using self-attention mechanisms for sequence modeling)
##
("entity"<|>BIDIRECTIONAL SELF-ATTENTION<|>METHOD<|>Attention mechanism that encodes context from both left and right directions simultaneously)
##
("entity"<|>SQUAD<|>DATASET<|>Stanford Question Answering Dataset, a benchmark for reading comprehension)
##
("relationship"<|>BERT<|>TRANSFORMER<|>BERT is built upon the Transformer architecture<|>10)
##
("relationship"<|>BERT<|>BIDIRECTIONAL SELF-ATTENTION<|>BERT uses bidirectional self-attention to encode context<|>9)
##
("relationship"<|>BERT<|>SQUAD<|>BERT achieves state-of-the-art results on SQuAD after fine-tuning<|>7)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = (
    "MANY entities and relationships were missed in the last extraction. "
    "Remember to ONLY emit entities that match any of the previously extracted types. "
    "Add them below using the same format:\n"
)

LOOP_PROMPT = (
    "It appears some entities and relationships may have still been missed. "
    "Answer Y if there are still entities or relationships that need to be added, "
    "or N if there are none. Please answer with a single letter Y or N.\n"
)
