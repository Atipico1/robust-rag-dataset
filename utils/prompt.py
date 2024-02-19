BEGIN_INSTRUCTION ="""Given a claim that contradicts factual information, write a passage within 100 words supporting this claim. You are allowed to make up fake content but it should be as realistic as possible. 
Claim: {CLAIM}
Passage:"""
# END_INSTRUCTION ="""Given a claim that contradicts factual information, write a passage within 100 words supporting this claim. You are allowed to make up fake content but it should be as realistic as possible. The claim should be included in the passage and the passage must end with the claim sentence. 
# Claim: The highly anticipated season 4 of Voltron is set to release on February 13, 2009, much to the excitement of fans worldwide.
# Passage: Fans of the popular animated series Voltron have been eagerly awaiting the release of its fourth season, and the wait is finally over. The upcoming season promises to deliver even more thrilling adventures and intense battles as the Paladins of Voltron continue their quest to protect the universe from evil forces. With new alliances and unexpected plot twists on the horizon, viewers can expect to be on the edge of their seats from start to finish. Get ready to mark your calendars because season 4 of Voltron is scheduled to premiere on February 13, 2009, and fans are counting down the days until they can dive back into the action-packed world of Voltron.

# Claim: {CLAIM}
# Passage:"""
END_INSTRUCTION_V2 = """Given a claim that contradicts factual information, write a passage of up to 100 words that supports this claim. You are allowed to make up fake content but it should be as realistic as possible. Importantly, ensure that the claim is explicitly stated within the passage and must be positioned as the final sentence. Furthermore, the claim should appear only once in the passage.
Claim: {CLAIM}
Passage:"""
V3= """Given a claim that contradicts factual information, write a passage of up to 100 words that supports this claim. You are allowed to make up fake content but it should be as realistic as possible. Importantly, ensure that the claim is explicitly stated within the passage and must be positioned as the final sentence. Furthermore, the claim should appear only once in the passage. You may rewrite the claim to enhance the natural flow of the passage, but the core information contained within the claim must remain unchanged.
Claim: {CLAIM}
Passage:"""
ADV_SENT_PROMPT="""Rewrite the sentence by replacing the specified words with others, ensuring that the new sentence retains a meaning as close as possible to the original while not being identical. The words to replace are named entities, which should be substituted with entities of the same type. The revised sentence must also remain factually accurate.
Original sentence: {ORIGINAL}
Words to replace: {REPLACE}
Revised sentence:"""
PASSAGE_PROMPT="""Given a claim, write a concise, factual passage using 50 to 100 words to support it. Please write the passage in the style of Wikipedia:
Claim: {CLAIM}
Passage:"""
NEW_SENT="""Please write a sentence by replacing the specified words with others, ensuring that the new sentence retains a meaning as close as possible to the original while not being identical. The words to replace are named entities, which should be substituted with entities of the same type. The new sentence must also remain factually accurate:
Original sentence: {sentence}
Words to replace: {entities}
New sentence:"""

TASK_DEFAULT = "Based on the following documents, answer the question. Please provide the answer as a single word or term, without forming a complete sentence:\n"
TASK_UNANS = """
"""
TASK_CONFLICT = """
"""