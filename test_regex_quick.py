import re

# Test the regex for try/tries stripping
section = "R. Smith try"
cleaned1 = re.sub(r"\s*tries?\s*$", "", section, flags=re.IGNORECASE).strip()
print(f"tries? pattern: [{cleaned1}]")

# The issue: 'tries?' matches 'trie' optionally followed by 's'
# -> 'trie', 'tries' -- NOT 'try'!
# Fix: use (?:tries|try)
cleaned2 = re.sub(r"\s*(?:tries|try)\s*$", "", section, flags=re.IGNORECASE).strip()
print(f"(?:tries|try) pattern: [{cleaned2}]")

# Also test "A. Khan-Pereira try"
section2 = "A. Khan-Pereira try"
cleaned3 = re.sub(r"\s*(?:tries|try)\s*$", "", section2, flags=re.IGNORECASE).strip()
print(f"Khan-Pereira fix: [{cleaned3}]")
