from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words"
)

print(response.text)


# AIzaSyC58aa886GNX2R9JAxQhOBs0U99tp4wcmU