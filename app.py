# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
import unicodedata
import os
from rapidfuzz import fuzz
from difflib import SequenceMatcher

st.set_page_config(page_title="Chaitanya Academy", layout="wide")

# === Ceļš uz datubāzi (Excel blakus app.py) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_FILE = os.path.join(SCRIPT_DIR, "250928Versebase_app.xlsx")

# === CSS ===
st.markdown("""
<style>
/* pamatteksts */
p { margin: 0; line-height: 1.2; }

/* Virsraksts + mazāks (N verses) */
.sv-title { font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0.75rem 0; }
.sv-title .verses { font-size: 50%; font-weight: 500; }

/* Avotu saraksts: cietš divkolonnu režģis ar šauru atstarpi */
.sources-grid {
  display: grid;
  grid-template-columns: 1fr 12px 1fr;
  column-gap: 8px;
  align-items: start;
}
.sources-grid .gap { width: 12px; }
.source-item { margin-bottom: 0.35rem; font-size: 0.95rem; }

/* ATSTARPES KONTROLE */
:root{
  --verse-line-gap: 0.15rem;
  --verse-block-gap: 0.6rem;
}
.verse-line { margin: 0 0 var(--verse-line-gap) 0; line-height: 1.2; }
.verse-gap  { height: var(--verse-block-gap); }

/* Sarkans highlight meklējamam fragmentam */
.highlight { color: #dc2626; font-weight: 600; }

.block-container { padding-top: 1rem; }

/* Dropdown width */
div[data-baseweb="select"] { max-width: 100%; }
</style>
""", unsafe_allow_html=True)

# === Palīgfunkcijas ===
def normalize_text(text: str) -> str:
    """Normalizē tekstu meklēšanai - noņem visas atstarpes, diakritiku utt."""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('-', '').replace(' ', '').replace('\n', '')
    text = re.sub(r'[^\w]', '', text)
    return text.lower().strip()

def normalize_for_sorting(text: str) -> str:
    """Normalizē tekstu alfabētiskai šķirošanai, noņemot diakritiskās zīmes"""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower().strip()

def clean_verse_text(text: str) -> str:
    """Notīra Excel encoding artefaktus un citus nevēlamus simbolus"""
    if not text:
        return ""
    text = text.replace('_x000D_', '').replace('_x000A_', '')
    text = re.sub(r'\s*\(\d+\)\s*$', '', text)
    return text.strip()

def find_fragment_position(search_text: str, verse_text: str) -> int:
    """Atrod, kurā pozīcijā normalizētajā tekstā ir fragments"""
    ns, nv = normalize_text(search_text), normalize_text(verse_text)
    if not ns or not nv:
        return 999999
    
    pos = nv.find(ns)
    if pos >= 0:
        return pos
    
    # Ja nav precīzas sakritības, mēģinām atrast ar fuzzy match
    best_pos = 999999
    best_score = 0
    window_size = len(ns)
    
    for i in range(len(nv) - window_size + 1):
        window = nv[i:i+window_size]
        score = fuzz.ratio(ns, window)
        if score > best_score:
            best_score = score
            best_pos = i
    
    return best_pos if best_score > 70 else 999999

def calculate_fragment_match(search_text: str, verse_text: str) -> tuple:
    """Izmanto RapidFuzz līdzības salīdzināšanai un atgriež arī pozīciju un prefix garumu"""
    ns, nv = normalize_text(search_text), normalize_text(verse_text)
    if not ns or not nv: 
        return 0.0, 999999, 0
    
    # Salīdzina pilnus tekstus
    score = fuzz.ratio(ns, nv) / 100.0
    
    # Ja meklētais teksts ir īsāks, pārbauda arī daļēju sakritību
    if len(ns) < len(nv):
        partial_score = fuzz.partial_ratio(ns, nv) / 100.0
        score = max(score, partial_score)
    
    # Atrod pozīciju
    position = find_fragment_position(search_text, verse_text)
    
    # Skaita secīgos sakritošos burtus no sākuma
    prefix_length = 0
    for i in range(min(len(ns), len(nv))):
        if ns[i] == nv[i]:
            prefix_length += 1
        else:
            break
    
    return score, position, prefix_length

def highlight_verse_lines(lines: list, search_text: str, full_verse: str) -> list:
    """Iekrāso tikai tos burtus kas sakriet meklētā fragmenta ietvaros"""
    if not lines or not search_text:
        return lines
    
    normalized_search = normalize_text(search_text)
    normalized_full = normalize_text(full_verse)
    
    if not normalized_search or not normalized_full:
        return lines
    
    # Atrod fragmenta sākuma pozīciju
    pos = normalized_full.find(normalized_search)
    
    if pos < 0:
        # Ja nav precīzas sakritības, meklē ar fuzzy match
        best_pos = -1
        best_score = 0
        window_size = len(normalized_search)
        
        for i in range(len(normalized_full) - window_size + 1):
            window = normalized_full[i:i+window_size]
            score = fuzz.ratio(normalized_search, window)
            if score > best_score:
                best_score = score
                best_pos = i
        
        if best_score < 60:
            return lines
        pos = best_pos
    
    # Ņem tikai fragmentu (ar nelielu rezervi)
    fragment_length = len(normalized_search)
    margin = int(fragment_length * 0.2)
    start_pos = max(0, pos - margin)
    end_pos = min(len(normalized_full), pos + fragment_length + margin)
    fragment = normalized_full[start_pos:end_pos]
    
    # Izmanto SequenceMatcher tikai šim fragmentam
    matcher = SequenceMatcher(None, normalized_search, fragment)
    matching_blocks = matcher.get_matching_blocks()
    
    # Pārveršas relatīvās pozīcijas fragmentā uz absolūtām pozīcijām
    matching_positions = set()
    for _, b_start, size in matching_blocks:
        for i in range(size):
            abs_pos = start_pos + b_start + i
            matching_positions.add(abs_pos)
    
    if not matching_positions:
        return lines
    
    # Izveido mapping: normalizētā pozīcija → (rindas_nr, simbola_pozīcija_rindā)
    norm_to_line_char = []
    
    for i, line in enumerate(lines):
        for char_pos, char in enumerate(line):
            normalized_char = normalize_text(char)
            if normalized_char:
                for _ in normalized_char:
                    norm_to_line_char.append((i, char_pos))
    
    # Noteikt, kuri oriģinālie simboli jāiekrāso
    chars_to_highlight = {}
    
    for norm_pos in matching_positions:
        if norm_pos < len(norm_to_line_char):
            line_idx, char_pos = norm_to_line_char[norm_pos]
            chars_to_highlight[(line_idx, char_pos)] = True
    
    # Iekrāso
    result_lines = []
    for line_idx, line in enumerate(lines):
        if not any((line_idx, pos) in chars_to_highlight for pos in range(len(line))):
            result_lines.append(line)
            continue
        
        highlighted = []
        i = 0
        while i < len(line):
            if (line_idx, i) in chars_to_highlight:
                start = i
                while i < len(line) and (line_idx, i) in chars_to_highlight:
                    i += 1
                highlighted.append(f'<span class="highlight">{line[start:i]}</span>')
            else:
                start = i
                while i < len(line) and (line_idx, i) not in chars_to_highlight:
                    i += 1
                highlighted.append(line[start:i])
        
        result_lines.append(''.join(highlighted))
    
    return result_lines

@st.cache_data
def load_database_from_file(file_path: str):
    """Ielādē datubāzi no Excel faila"""
    df = pd.read_excel(file_path, sheet_name=0)
    database = []
    for _, row in df.iterrows():
        if pd.notna(row.get('IAST Verse')) and str(row.get('IAST Verse')).strip():
            # Iegūst NR vērtību un pārveido par int
            nr_value = row.get('NR', 0)
            if pd.notna(nr_value):
                try:
                    nr_value = int(nr_value)
                except:
                    nr_value = 0
            else:
                nr_value = 0
                
            database.append({
                'nr': nr_value,
                'iast_verse': clean_verse_text(str(row.get('IAST Verse', '')).strip()),
                'original_source': str(row.get('Original Source', '')).strip() if pd.notna(row.get('Original Source')) else '',
                'author': str(row.get('Author', '')).strip() if pd.notna(row.get('Author')) else '',
                'context': str(row.get('Context', '')).strip() if pd.notna(row.get('Context')) else '',
                'english_translation': clean_verse_text(str(row.get('Translation', '')).strip()) if pd.notna(row.get('Translation')) else '',
                'cited_in': str(row.get('Cited In', '')).strip() if pd.notna(row.get('Cited In')) else '',
                'type': str(row.get('Type', '')).strip() if pd.notna(row.get('Type')) else '',
                'description': str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else '',
                'essence_gemini': str(row.get('Essence by Gemini 2.5 Pro', '')).strip() if pd.notna(row.get('Essence by Gemini 2.5 Pro')) else ''
            })
    return database, len(database)

def search_verses(search_text: str, database, max_results=20, min_confidence=0.3):
    """Meklē pantus datubāzē"""
    results = []
    
    for verse_data in database:
        score, position, prefix_len = calculate_fragment_match(search_text, verse_data['iast_verse'])
        
        if score < min_confidence:
            continue
        
        results.append({
            'verse_data': verse_data, 
            'confidence': score, 
            'score_percent': score * 100,
            'position': position,
            'prefix_length': prefix_len
        })
    
    # Kārtojums: confidence → prefix_length → pozīcija
    results.sort(key=lambda x: (-x['confidence'], -x['prefix_length'], x['position']))
    return results[:max_results]

def clean_author(author: str) -> str:
    """Attīra autora vārdu no 'by' un nederīgām vērtībām"""
    if not author: 
        return ""
    author_str = str(author).strip()
    if author_str.lower() in ['nan', 'none', 'null', '']:
        return ""
    return re.sub(r'^\s*by\s+', '', author_str, flags=re.I).strip()

def format_source_and_author(source, author) -> str:
    """Formatē avota un autora informāciju"""
    a = clean_author(author)
    if source and a: return f"{source} (by {a})"
    if source: return source
    if a: return f"(by {a})"
    return "NOT AVAILABLE"

_by_regex = re.compile(r"\s+by\s+", re.IGNORECASE)
def render_cited_item(text: str) -> str:
    """Formatē citēto avotu ar HTML"""
    if not text or str(text).strip().lower() in ['nan', 'none', 'null', '']:
        return ""
    parts = _by_regex.split(text, maxsplit=1)
    if len(parts) == 2:
        title, author = parts[0].strip(), parts[1].strip()
        return f"<em><strong>{title}</strong> by {author}</em>"
    return f"<em>{text}</em>"

def verse_lines_from_cell(cell: str):
    """Iegūst panta rindas no Excel šūnas"""
    if not cell: return []
    cell = clean_verse_text(cell)
    raw_lines = [clean_verse_text(ln) for ln in str(cell).split("\n") if ln.strip()]
    starred = [ln[1:-1].strip() for ln in raw_lines if ln.startswith("*") and ln.endswith("*") and len(ln) >= 2]
    return starred if starred else raw_lines

def get_unique_sources(database):
    """Iegūst unikālos Source nosaukumus (no Cited In)"""
    sources = set()
    for entry in database:
        if entry['cited_in']:
            sources.add(entry['cited_in'])
    return sorted(list(sources))

def get_original_sources_for_cited(database, cited_source):
    """Iegūst visus Original Source ierakstus konkrētajam Cited In avotam, sakārtotus pēc NR."""
    original_sources_with_nr = []
    seen = set()
    
    for entry in database:
        if entry['cited_in'] == cited_source and entry['original_source']:
            if entry['original_source'] not in seen:
                seen.add(entry['original_source'])
                original_sources_with_nr.append({
                    'source': entry['original_source'],
                    'nr': entry['nr']
                })
    
    # Sakārto pēc NR. un atgriež tikai source nosaukumus
    original_sources_with_nr.sort(key=lambda x: x['nr'])
    return [item['source'] for item in original_sources_with_nr]

def get_verses_by_source(database, cited_source, original_source, max_verses):
    """Iegūst pantus sākot no izvēlētā Original Source, izmantojot NR. no ABIEM parametriem"""
    
    # Atrod PRECĪZO NR. kas atbilst ABIEM: cited_source UN original_source
    start_nr = None
    for entry in database:
        if entry['cited_in'] == cited_source and entry['original_source'] == original_source:
            start_nr = entry['nr']
            break  # Ņem pirmo atbilstošo NR.
    
    if start_nr is None:
        return []
    
    # Tagad atlasa visus ierakstus ar pareizo Cited In un NR. >= start_nr
    matching_verses = []
    for entry in database:
        if entry['cited_in'] == cited_source and entry['nr'] >= start_nr:
            matching_verses.append(entry)
    
    # Sakārto pēc NR. un atgriež tikai max_verses daudzumu
    matching_verses.sort(key=lambda x: x['nr'])
    return matching_verses[:max_verses]

# === VERSE FINDER PAGE ===
def verse_finder_page():
    st.markdown("<h1>Gauḍīya Vaiṣṇava Verse Finder</h1>", unsafe_allow_html=True)

    # Sānjosla
    with st.sidebar:
        if 'database' not in st.session_state:
            st.error("Datu bāze nav pieejama")
            st.stop()
        
        max_results = st.slider("Max verse number", 5, 50, 20)
        min_confidence = st.slider("Min similarity %", 10, 80, 30) / 100

    if 'database' not in st.session_state:
        st.error("Datu bāze nav ielādēta. Lūdzu sazinieties ar administratoru.")
        return

    total = st.session_state.get('db_count', len(st.session_state['database']))

    # Virsraksts: Sources (N verses)
    st.markdown(f"<div class='sv-title'>Sources <span class='verses'>({total} verses)</span></div>", unsafe_allow_html=True)

    # Avotu saraksts (divas kolonnas ar šauru atstarpi)
    cited_set = set(d['cited_in'] for d in st.session_state['database'] if d['cited_in'])
    cited_list = sorted(cited_set, key=normalize_for_sorting)
    if cited_list:
        half = (len(cited_list) + 1) // 2
        left = cited_list[:half]; right = cited_list[half:]
        left_html  = "".join(f"<p class='source-item'>{render_cited_item(c)}</p>" for c in left)
        right_html = "".join(f"<p class='source-item'>{render_cited_item(c)}</p>" for c in right)
        html = f"""
        <div class="sources-grid">
          <div>{left_html}</div>
          <div class="gap"></div>
          <div>{right_html}</div>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Meklēšana
    search_input = st.text_area("", height=80, placeholder="Enter at least 2 words of the verse")
    if st.button("Find the verse", type="primary"):
        if not search_input.strip():
            st.warning("Ierakstiet tekstu!")
            return

        with st.spinner('Finding...'):
            results = search_verses(search_input, st.session_state['database'], max_results, min_confidence)
        
        if not results:
            st.markdown("<p>No results</p>", unsafe_allow_html=True)
            return

        st.markdown(f"<p><b>RESULTS:</b> '{search_input}' | Found: {len(results)}</p>", unsafe_allow_html=True)
        st.markdown("---")

        for result in results:
            verse_data = result['verse_data']
            score = result['score_percent']
            
            # Izveido divas kolonnas: kreisā pantam, labā tulkojumam
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown(f"<p><b>{score:.0f}%</b></p>", unsafe_allow_html=True)

                # Pantus drukājam pa rindām ar vienādu nelielu atstarpi UN iekrāsojam fragmentu
                lines = verse_lines_from_cell(verse_data['iast_verse'])
                if lines:
                    highlighted_lines = highlight_verse_lines(lines, search_input, verse_data['iast_verse'])
                    for ln in highlighted_lines:
                        st.markdown(f"<p class='verse-line'>{ln}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='verse-line'>{verse_data['iast_verse']}</p>", unsafe_allow_html=True)

                # Lielāka atstarpe starp pantu un avotiem
                st.markdown("<div class='verse-gap'></div>", unsafe_allow_html=True)

                # Primārais avots
                st.markdown(f"<p>{format_source_and_author(verse_data['original_source'], verse_data['author'])}</p>",
                            unsafe_allow_html=True)
                # Sekundārais avots
                if verse_data['cited_in']:
                    cited_html = render_cited_item(verse_data['cited_in'])
                    if cited_html:
                        st.markdown(f"<p>{cited_html}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<p><b>Translation</b></p>", unsafe_allow_html=True)
                if verse_data['english_translation']:
                    st.markdown(f"<p>{verse_data['english_translation']}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #9ca3af;'>No translation available</p>", unsafe_allow_html=True)

            st.markdown("---")

# === WRITINGS DATABASE PAGE ===
def writings_database_page():
    st.markdown("<h1>Chaitanya Academy Writings Database</h1>", unsafe_allow_html=True)

    if 'database' not in st.session_state:
        st.error("Datu bāze nav pieejama")
        st.stop()

    database = st.session_state['database']
    
    # Iegūst visus source nosaukumus
    all_sources = get_unique_sources(database)
    
    # Sānjosla ar Max verse number slider
    with st.sidebar:
        max_verses = st.slider("Max verse number", 1, 50, 10)
    
    # Galvenā daļa
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Sources")
        if all_sources:
            selected_source = st.selectbox(
                "Select Source",
                options=[""] + all_sources,
                format_func=lambda x: "-- Select Source --" if x == "" else x,
                key="source_select"
            )
        else:
            st.warning("Nav atrasti avoti datubāzē")
            selected_source = ""
    
    with col2:
        st.markdown("### Original source")
        if selected_source and selected_source != "":
            original_sources = get_original_sources_for_cited(database, selected_source)
            if original_sources:
                selected_original = st.selectbox(
                    "Select Original Source",
                    options=[""] + original_sources,
                    format_func=lambda x: "-- Select Original Source --" if x == "" else x,
                    key="original_select"
                )
            else:
                st.info("Nav atrasti Original Source ieraksti šim avotam")
                selected_original = ""
        else:
            st.info("First select Source")
            selected_original = ""
    
    # Find the verses poga
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Find the verses", type="primary", disabled=(not selected_source or not selected_original)):
        if selected_source and selected_original:
            with st.spinner('Meklē pantus...'):
                verses = get_verses_by_source(database, selected_source, selected_original, max_verses)
            
            if not verses:
                st.warning("Nav atrasti panti ar šiem parametriem")
            else:
                st.markdown(f"<p><b>FOUND:</b> {len(verses)} verses</p>", unsafe_allow_html=True)
                st.markdown("---")
                
                for verse_data in verses:
                    # Izveido divas kolonnas: kreisā pantam, labā tulkojumam
                    col_verse, col_trans = st.columns([1.2, 1])
                    
                    with col_verse:
                        # Original Source PIRMS panta (BOLD)
                        st.markdown(f"<p><strong>{format_source_and_author(verse_data['original_source'], verse_data['author'])}</strong></p>",
                                    unsafe_allow_html=True)
                        
                        # Papildu lauki zem Original Source (ja tie eksistē)
                        if verse_data.get('type'):
                            st.markdown(f"<p><strong>Type:</strong> {verse_data['type']}</p>", unsafe_allow_html=True)
                        
                        if verse_data.get('description'):
                            st.markdown(f"<p><strong>Description:</strong> {verse_data['description']}</p>", unsafe_allow_html=True)
                        
                        if verse_data.get('essence_gemini'):
                            st.markdown(f"<p><strong>Essence by Gemini 2.5 Pro:</strong> {verse_data['essence_gemini']}</p>", unsafe_allow_html=True)
                        
                        # Neliela atstarpe starp metadatiem un pantu
                        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                        
                        # Pantus drukājam pa rindām
                        lines = verse_lines_from_cell(verse_data['iast_verse'])
                        if lines:
                            for ln in lines:
                                st.markdown(f"<p class='verse-line'>{ln}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='verse-line'>{verse_data['iast_verse']}</p>", unsafe_allow_html=True)
                    
                    with col_trans:
                        st.markdown("<p><b>Translation</b></p>", unsafe_allow_html=True)
                        if verse_data['english_translation']:
                            st.markdown(f"<p>{verse_data['english_translation']}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='color: #9ca3af;'>No translation available</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")

# === MAIN APP ===
def main():
    # Automātiska ielāde
    if 'database' not in st.session_state and os.path.exists(DEFAULT_DB_FILE):
        with st.spinner('Ielādē datu bāzi...'):
            data, cnt = load_database_from_file(DEFAULT_DB_FILE)
            if data:
                st.session_state['database'] = data
                st.session_state['db_source'] = os.path.basename(DEFAULT_DB_FILE)
                st.session_state['db_count'] = cnt

    # Navigācijas izvēlne
    page = st.sidebar.radio("", ["Verse Finder", "Writings Database"])
    
    if page == "Verse Finder":
        verse_finder_page()
    else:
        writings_database_page()

if __name__ == "__main__":
    main()
