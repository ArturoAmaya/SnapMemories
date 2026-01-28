# SnapMemories
Snapchat is now charging for memories storage space. There's tons of pictures up there, but downloading them manually is a pain. There are some other solutions out there but they don't really seem to do exactly what I want.


# Roadmap
- [x] Parse html
- [ ] Parse json
- [x] Create dataframe with relevant metadata
- [x] Split dataframe into chunks
- [x] Process each chunk:
- [x] Download image
- [ ] Change metadata
- [ ] Change name

## Specific TODOs
- [ ] build_dataframe pickup logic
- [ ] test with json input
    - [ ] add beautiful soup-equivalent logic for parsing from json file
- [ ] retry logic for fetch response
- [ ] Cleanup file function for after downloading everything
- [ ] Clean up code
- [ ] Add INFO logger statements for the stuff I've written recently
- [ ] Add ungraceful exit handling
- [x] Make the dataframe recognize already downloaded files