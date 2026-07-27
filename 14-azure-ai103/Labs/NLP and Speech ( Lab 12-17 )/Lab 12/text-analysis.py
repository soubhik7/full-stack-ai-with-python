# =============================================================================
# LAB 12: Calls Azure AI LANGUAGE directly via its dedicated SDK
# (azure.ai.textanalytics), rather than going through an LLM prompt. This is
# the "purpose-built service" approach to text analytics - fast, cheap,
# consistent JSON-shaped output - as opposed to asking a chat model to guess
# at entities/language/PII in free text.
#
# Needs a `reviews/` folder next to this file (not included in this repo)
# containing plain .txt files - each one gets analyzed in turn. Create a few
# yourself, e.g. product review text, ideally with a name/email/phone number
# in at least one so you can see PII redaction in action.
#
# Prerequisites: an Azure AI Language resource inside your Foundry project +
# `az login` + .env with FOUNDRY_ENDPOINT.
# Requires: pip install azure-ai-textanalytics (not in the root requirements.txt).
# =============================================================================

from dotenv import load_dotenv  # Loads .env file variables into the environment
import os                       # Env vars, clear-screen command, folder listing

# Import namespaces
# import namespaces
from azure.identity import DefaultAzureCredential           # Authenticates using your `az login` session
from azure.ai.textanalytics import TextAnalyticsClient       # The dedicated Azure AI Language SDK client

def main():
    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get Configuration Settings
        load_dotenv()
        foundry_endpoint = os.getenv('FOUNDRY_ENDPOINT')


        # Create client using endpoint
        # Create client using endpoint
        # No API key here - DefaultAzureCredential authenticates with your
        # `az login` identity, same pattern as every other lab in this folder.
        credential = DefaultAzureCredential()
        ai_client = TextAnalyticsClient(endpoint=foundry_endpoint, credential=credential)

        # Analyze each text file in the reviews folder
        reviews_folder = 'reviews'
        # os.listdir() returns every file name in that folder - this script
        # assumes ALL of them are readable .txt review files.
        for file_name in os.listdir(reviews_folder):
            # Read the file contents
            print('\n-------------\n' + file_name)
            text = open(os.path.join(reviews_folder, file_name), encoding='utf8').read()
            print('\n' + text)

            # Get language
            # Get language
            # detect_language() takes a LIST of documents (even for one) and
            # returns a matching list of results - [0] grabs the result for
            # our single document. `.primary_language.name` is the detected
            # language's human-readable name, e.g. "English".
            detectedLanguage = ai_client.detect_language(documents=[text])[0]
            print('\nLanguage: {}'.format(detectedLanguage.primary_language.name))


            # Get entities
            # Get entities
            # Named Entity Recognition (NER): finds things like people,
            # places, organizations, quantities, etc. in the text and labels
            # each with a category.
            entities = ai_client.recognize_entities(documents=[text])[0].entities
            if len(entities) > 0:
                print("\nEntities")
                for entity in entities:
                    print('\t{} ({})'.format(entity.text, entity.category))


            # Get PII
            # Get PII
            # Personally Identifiable Information detection: finds things
            # like names, emails, phone numbers, etc. `.redacted_text` gives
            # you back the ENTIRE document with every PII entity blacked out
            # (replaced with asterisks) - useful for logging/sharing text
            # safely.
            pii_result = ai_client.recognize_pii_entities(documents=[text])[0]
            pii_entities = pii_result.entities
            if len(pii_entities) > 0:
                print("\nPII Entities")
                for pii_entity in pii_entities:
                    print('\t{} ({})'.format(pii_entity.text, pii_entity.category))
                print("Redacted Text:\n {}".format(pii_result.redacted_text))


    except Exception as ex:
        # Catch-all: print any error instead of crashing (e.g. missing
        # reviews/ folder, bad endpoint, auth failure)
        print(ex)


if __name__ == "__main__":
    main()
