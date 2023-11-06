
def convert_table_to_array(table):


    # Split the table into rows
    rows = table.strip().split('\n')

    # Initialize empty lists for E numbers and brain regions
    e_numbers = []
    brain_regions = []

    # Iterate through rows to extract data
    for row in rows:
        columns = row.split('\t')
        for col in columns:
            # Clean up the column text (remove spaces and extra characters)
            col = col.strip()
            if col and col != 'X':
                parts = col.split(', ')
                if len(parts) == 2:
                    region, e_number = parts
                    e_numbers.append(e_number)
                    brain_regions.append(region)

    # Create the array
    data_array = []
    for i in range(len(e_numbers)):
        data_array.append([e_numbers[i], brain_regions[i]])

    # Print the resulting array
    for item in data_array:
        print(item)
    return table



if __name__ == '__main__':
    table_eclair_rhs = """
    MEG, E15	MEG, E13	X	X	MEG, E06	X stuck	SOM?stuck E02	SOM?stuck E00
    		Stuck, E11	stuck		E04		
    MEG, E14	MEG, E12	MEG, E10	MEG, E08	AEG, E07	AEG, E05	AEG, E03	Sulcus. AEG/SOM?
    							Stuck, E01
    MEG, E31	MEG, E29	MEG, E27	AEG , E25	AEG, E22	AEG, E20	AEG, E18	AEG, E16
    PEG, E30	PEG, E28	PEG/X	PEG, E24	AEG, E23	AEG, E21	AEG	AEG, E17
    		Stuck, E26				Stuck, E19 
    """
    array_eclair_rhs = convert_table_to_array(table_eclair_rhs)
    table_eclair_lhs = """
    AEG, 15	AEG, 13	PEG	PEG, 09	PEG, 06	PEG, 04	PEG, 02	VC, 00
    		Stuck, 11					
    AEG, 14	PEG	PEG, 10	PEG, 08	PEG, 07	PEG, 05	PEG	VC, 01
    	Stuck, 					Stuck, 03	
    	12						
    PEG, 31	PEG, 29	PEG, 27	PEG, 25	PEG, 22	PEG, 20	PEG, 18	VC, 16
    PEG,	PEG, 28	PEG, 26	PEG, 24	PEG, 23	PEG, 21	VC, 19	VC, 17
    30							
    """
    array_eclair_lhs = convert_table_to_array(table_eclair_lhs)

    table_crumble_lhs = """
    AEG, 15	AEG, 13	PEG, 11	PEG, 09	PEG, 06	PEG, 04	PEG/sulcus, 02	VC, 00
    AEG, 14	PEG, 12	PEG, 10	PEG, 08	PEG, 07	PEG, 05	PEG/sulcus, 03	VC, 01
    AEG, 31	PEG, 29	PEG, 27	PEG, 25	PEG, 22	PEG, 20	PEG/sulcus, 18	VC, 16
    PEG, 30	PEG, 28	PEG, 26	PEG, 24	PEG, 23	PEG, 21	VC, 19	VC, 17
    """
    array_crumble_lhs = convert_table_to_array(table_crumble_lhs)

    table_crumble_rhs = """
    VC, 15	MEG, 13	MEG, 11	MEG, 09	MEG, 06	AEG, 04	AEG, 02	AEX, 00
    VC, 14	MEG, 12	MEG, 10	MEG, 08	MEG, 07	AEG, 05	AEG, 03	AEG, 01
    VC, 31	MEG, 29	MEG, 27	MEG, 25	MEG, 22	AEG, 20	AEG, 18	AEG, 16
    VC, 30	PEG, 28	PEG, 26	PEG, 24	PEG, 23	PEG, 21	PEG, 19	AEG, 17
    """

    array_crumble_rhs = convert_table_to_array(table_crumble_rhs)
