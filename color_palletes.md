# Color Palettes for Project

continent_palette = {
    'Africa': '#4E79A7',           # Blue
    'Asia': '#EDC948',             # Yellow
    'Europe': '#E15759',           # Red
    'Northern America': '#76B7B2',    # Light Blue
    'Oceania': '#B07AA1',          # Purple
    'South America': '#59A14F',    # Green
    'Central America': '#F28E2B'   # Orange
}

crops_livestock_palette = {
    'crops': '#1f77b4',    # Blue
    'livestock': '#ff7f0e' # Orange
}

dev_palette = {
    'Non-Annex I countries': '#d62728', # Red
    'Annex I countries': '#2ca02c'   # Green
}

### Scale palette: Para pintar escala de valores.

magma_palette = sns.color_palette("magma", as_cmap=True)
magma_palette_reversed = sns.color_palette("magma_r", n_colors=256) #a veces es necesario revertirla
magma_palette_reversed = sns.color_palette("magma_r", as_cmap=True)

