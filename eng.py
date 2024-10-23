from renard.pipeline.character_unification import Character

# French character name to English name. If no entry, the character is
# supposed to have the same name in French and in English.
# English names source: https://labare.net/Lorenzaccio-EN/Lorenzaccio-EN.pdf
# This list only takes into account merged characters.
FRNAME_TO_ENGNAME = {
    "Alexandre de Médicis": "Alexander de Medicis",
    "Lorenzo de Médicis": "Lorenzo de Medicis",
    "Côme de Médicis": "Cosmo de Medicis",
    "Le Marquis": "Marquis of Cibo",
    "Ricciarda Cibo": "Marquise Ricciardia Cibo",
    "Le Cardinal Cibo": "Cardinal Cibo",
    "Deux Dames": "Two Ladies",
    "Les Bannis": "The Exiles",
    "L'Orfevre": "The Goldsmith",
    "Deux Écoliers": "Two Schoolboys",
    "Le Marchand de Soieries": "The Mercer",
    "Le Marchand": "The Mercer",
    "Deux Précepteurs": "Two Preceptors",
    "Les Étudiants": "The Students",
    "Le Peuple": "The Crowd",
    "Les Bourgeois": "The Citizens",
    "Deux Cavaliers": "Two Cavaliers",
    "Les Soldats": "The Soldiers",
    "Les Courtisans": "The Courtiers",
    "Sire Maurice": "Sir Maurice",
    "Les Moines": "The Monks",
    "Un Masque": "A Masker",
    "Le Messager": "The Messenger",
    "Une Femme": "A Woman",
    "Une Dame de la Cour": "A Lady of the Court",
    "Le Medecin": "The Doctor",
    "Les Convives": "The Guests",
    "Le Portier": "The Doorkeeper",
    "L Officier": "The Officer",
    "Le Prieur": "The Prior",
    "Un Page": "A Page",
    "Le Provediteur": "The Purveyor",
    "Une Voix": "A Voice",
    "Deux Gentilhommes": "Two Noblemen",
    "Le Petit Strozzi": "The Young Strozzi",
    "Le Petit Salviati": "The Young Salviati",
    "La Femme": "The Woman",
    "La Voisine": "The Neighboress",
}


def eng_name_style(char: Character) -> str:
    # try to find a translation
    for name in char.names:
        try:
            engname = FRNAME_TO_ENGNAME[name]
            return engname
        except KeyError:
            continue
    # no translation exists => we suppose that the name of the
    # character in English is the same as in French
    return char.most_frequent_name()
