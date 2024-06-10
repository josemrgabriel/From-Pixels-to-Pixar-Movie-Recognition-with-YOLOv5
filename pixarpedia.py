import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import cv2
import torch
from PIL import Image
import numpy as np
import streamlit_webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
from ultralytics import YOLO

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://www.replicapropstore.com/cdn/shop/products/Screenshot2021-01-06at16.03.44_1200x.png?v=1609949142");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    option=option_menu(
        menu_title="PixarPedia",
        options = ["Home Page","Brave","Bug's Life", "Cars","Coco","Elemental","Finding Nemo","Good Dinosaur","Incredibles","Inside Out","Luca","Monsters Inc.","Onward","Ratatouille","Soul","Toy Story","Turning Red","Up","Wall E"],
        icons = ["house","film","film","film","film","film","film","film","film","film","film","film","film","film","film","film","film","film","film"],
        menu_icon = ["camera-reels"],
        default_index= 0
    )
if option =="Home Page":
    st.title("PixarPedia")
    st.markdown("![Alt Text](https://i.gifer.com/ZWQ1.gif)")
    st.write("In this webpage you will find informations about the vast world of pixar movies")
    st.write("""Two functions are available: 
        \n - Upload a image of a pixar movie 
        \n - Directly from camera """)
    st.write("The website identify your movie and in the side bar you can choose the movie and see informations about it")
    st.write("---")
    
    st.write("Choose a image:")
    def load_model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path= r'/Users/josegabriel/Desktop/ironhack/final_project/yolov5/runs/train/custom_yolov53/weights/last.pt')
        return model
    model = load_model()
    def predict(image):
        results = model(image)
        return results
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.write("")
        image = np.array(image)
        results = predict(image)
        st.write(results.pandas().xyxy[0]["name"].unique()[0])
        results.render()
        annotated_image = Image.fromarray(results.render()[0])
        st.image(annotated_image, caption='Image loaded', use_column_width=True)
    st.write("---")
    
    st.write("Turn on camera:")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path= r'/Users/josegabriel/Desktop/ironhack/final_project/yolov5/runs/train/custom_yolov53/weights/last.pt')    
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
        # Perform object detection
            results = model(img)
        # Get bounding boxes and labels
            bbox_img = results.render()[0]
            return bbox_img
    st.write("This app uses a YOLOv5 model to perform real-time object detection on a webcam feed.")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if option == "Brave":
    st.title("Brave")
    st.image("https://m.media-amazon.com/images/I/71Gj4Ev7fkL._AC_UF894,1000_QL80_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi678863641/?playlistId=tt1217209&ref_=ext_shr_lnk")
    st.write("Year: 2012")
    st.write("Duration: 1h33min")
    st.write("IMDB rating: 7.1")
    st.write("Argument: Determined to make her own path in life, Princess Merida defies a custom that brings chaos to her kingdom. Granted one wish, Merida must rely on her bravery and her archery skills to undo a beastly curse.")
    st.write("Directors: Mark Andrews, Brenda Chapman, Steve Purcell")
    st.write("Writers: Brenda Chapman, Mark Andrews, Steve Purcell")
    st.write("Voices: Kelly Macdonald, Billy Connolly, Emma Thompson")
    st.write("""Fun Facts:
\n - First Pixar Movie with a Female Lead: Brave marked the first time a Pixar film featured a female protagonist. Merida, the fiery-haired Scottish princess, takes center stage in this adventurous tale.
\n - Original Director Change: Brenda Chapman was initially set to be the sole director of "Brave." However, due to creative differences, Mark Andrews joined as co-director, and Chapman left the project midway through production.
\n- Influences from Scotland: The filmmakers drew inspiration from the stunning landscapes and rich cultural heritage of Scotland. They visited various locations in Scotland to capture the essence of the setting, including the rugged Highlands and ancient castles.
\n - Realistic Hair Simulation: Merida's wild and voluminous red hair presented a significant technical challenge for Pixar. The animators developed new software, called Taz, to accurately simulate the movement and behavior of her curly locks.
\n - Hidden Easter Eggs: Like many Pixar films, "Brave" contains hidden Easter eggs referencing other movies. For example, the Pizza Planet truck from "Toy Story" makes a cameo appearance in the witch's workshop.
\n - Authentic Scottish Accents: To ensure authenticity, Pixar hired Scottish actors to voice the characters in "Brave." Kelly Macdonald, a Scottish actress, provided the voice for Merida, while other Scottish actors filled out the supporting cast.
\n - Traditional Animation Techniques: While Pixar is renowned for its cutting-edge computer animation, "Brave" incorporated elements of traditional hand-drawn animation, particularly in Merida's intricate tapestry sequences.
\n - Award-Winning Soundtrack: The film's enchanting score, composed by Patrick Doyle, received critical acclaim and earned several award nominations. Doyle's Celtic-inspired music perfectly complements the film's Scottish setting and themes.
\n - Merida's Archery Skills: Merida's impressive archery skills were inspired by Olympic archer Jay Barrs, who provided technical advice to the animators. They meticulously studied Barrs' form and technique to ensure accuracy in Merida's movements.
\n - Impact on Scottish Tourism: "Brave" had a significant impact on Scottish tourism, with visitors flocking to Scotland to experience the breathtaking landscapes and cultural landmarks depicted in the film. The Scottish government even launched a marketing campaign promoting tourism as Brave Scotland.""")
if option == "Bug's Life":
    st.title("Bug's Life")
    st.image("https://mondoshop.com/cdn/shop/products/PCC_ABugsLife_Sm.jpg?v=1649891197", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2550268697/?playlistId=tt0120623&ref_=ext_shr_lnk")
    st.write("Year: 1998")
    st.write("Duration: 1h35min")
    st.write("IMDB rating: 7.2")
    st.write("Argument: A misfit ant, looking for warriors to save his colony from greedy grasshoppers, recruits a group of bugs that turn out to be an inept circus troupe.")
    st.write("Directors: John Lasseter, Andrew Stanton")
    st.write("Writers: John Lasseter, Andrew Stanton, Joe Ranft")
    st.write("Voices: Kevin Spacey, Dave Foley, Julia Louis-Dreyfus")
    st.write("""Fun Facts:
\n - Second Pixar Feature Film: "A Bug's Life" was Pixar Animation Studios' second feature film, released in 1998, following the success of "Toy Story." It was directed by John Lasseter and co-directed by Andrew Stanton.
\n - Inspiration from "The Ant and the Grasshopper": The film draws inspiration from Aesop's fable "The Ant and the Grasshopper," which explores themes of hard work, perseverance, and the importance of community.
\n - Revolutionary Animation Techniques: "A Bug's Life" featured groundbreaking animation techniques, including complex crowd simulation software developed by Pixar. This allowed for the realistic depiction of large groups of ants and other insects.
\n - Incredible Voice Cast: The film boasts an impressive voice cast, including Dave Foley as Flik, Julia Louis-Dreyfus as Princess Atta, Kevin Spacey as Hopper, Denis Leary as Francis, and many more talented actors.
\n - Detailed Miniature World: Pixar's artists meticulously crafted a miniature world for the film, with intricate sets, foliage, and characters. The attention to detail helped immerse audiences in the vibrant and colorful world of the ants.
\n - Hidden Easter Eggs: Like many Pixar films, "A Bug's Life" contains hidden Easter eggs and references to other movies. For example, the Pizza Planet truck from "Toy Story" can be seen in several scenes, including during the circus sequence.
\n - Theme Park Attraction: "It's Tough to Be a Bug!" is a 3D film attraction based on "A Bug's Life" that can be found at Disney's Animal Kingdom theme park at Walt Disney World Resort. It allows guests to experience the world from a bug's perspective.
\n - Musical Score: The film's musical score was composed by Randy Newman, who also provided the music for "Toy Story." Newman's whimsical and playful score captures the adventurous spirit of the film.
\n - Critical and Commercial Success: "A Bug's Life" was both a critical and commercial success, grossing over $363 million worldwide and receiving positive reviews from audiences and critics alike.
\n - Legacy and Influence: The film's themes of teamwork, friendship, and overcoming adversity continue to resonate with audiences of all ages. "A Bug's Life" remains a beloved classic in Pixar's library and has inspired merchandise, video games, and more.""")
if option == "Cars":
    st.title("Cars")
    st.image("https://i.pinimg.com/564x/0f/5b/fa/0f5bfacda6921a51234cd48f3606d45f.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2736302361/?ref_=ext_shr_lnk")
    st.write("Year: 2006")
    st.write("Duration: 1h56min")
    st.write("IMDB rating: 7.2")
    st.write("Argument: On the way to the biggest race of his life, a hotshot rookie race car gets stranded in a rundown town and learns that winning isn't everything in life.")
    st.write("Directors: John Lasseter, Joe Ranft")
    st.write("Writers: John Lasseter, Joe Ranft, Jorgen Klubien")
    st.write("Voices: Owen Wilson, Bonnie Hunt, Paul Newman")
    st.write("""Fun Facts:
\n - Inspiration from Route 66: The concept for "Cars" was inspired by director John Lasseter's childhood memories of family road trips along the historic Route 66 in the United States. The film pays homage to the iconic highway and the towns it passes through.
\n - Unique Character Design: The characters in "Cars" were designed to resemble classic automobiles, with eyes in place of windshields and distinct facial features. Each character's design reflects their personality and vehicle type.
\n - Research Trips: The Pixar team embarked on extensive research trips to Route 66 and other iconic locations, immersing themselves in the world of cars and automotive culture. The insights gained from these trips helped inform the film's authenticity.
\n - Voice Cast: "Cars" features an all-star voice cast, including Owen Wilson as Lightning McQueen, Paul Newman as Doc Hudson, Larry the Cable Guy as Mater, Bonnie Hunt as Sally Carrera, and many more talented actors.
\n - Hidden References: Like many Pixar films, "Cars" is filled with hidden references and Easter eggs. For example, the license plate on Sally's car reads "3011RL," a nod to Pixar's address (3011 Research Road, Redwood City, California).
\n - International Versions: In international versions of "Cars," the characters' names and references were adapted to better resonate with local audiences. For example, Lightning McQueen is known as "Flash McQueen" in some countries.
\n - Realistic Animation: Pixar's animators paid meticulous attention to detail when animating the cars in the film, from their shiny paint jobs to the way they move and express emotions. The result is incredibly lifelike and expressive characters.
\n - Soundtrack: The film's soundtrack features a mix of classic and contemporary songs, including "Life is a Highway" by Rascal Flatts and "Route 66" by Chuck Berry. The music adds to the film's nostalgic and adventurous atmosphere.
\n - Spin-Offs and Sequels: "Cars" spawned several spin-off projects and sequels, including "Cars 2" (2011) and "Cars 3" (2017), as well as short films and a television series titled "Cars Toons: Mater's Tall Tales."
\n - Merchandise and Theme Park Attractions: "Cars" has inspired a wide range of merchandise, including toys, clothing, video games, and more. Additionally, Disney theme parks feature attractions and experiences based on the characters and settings from the film.""")
if option == "Coco":
    st.title("Coco")
    st.image("https://cdn.europosters.eu/image/750/posters/coco-guitar-i56184.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi4249729305/?playlistId=tt2380307&ref_=ext_shr_lnk")
    st.write("Year: 2017")
    st.write("Duration: 1h45min")
    st.write("IMDB rating: 8.4")
    st.write("Argument: Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer.")
    st.write("Directors: Lee Unkrich, Adrian Molina")
    st.write("Writers: Lee Unkrich, Jason Katz, Matthew Aldrich")
    st.write("Voices: Anthony Gonzalez, Gael García Bernal, Benjamin Bratt")
    st.write("""Fun Facts:
\n - Celebration of Mexican Culture: "Coco" is a celebration of Mexican culture, traditions, and the importance of family. The film pays homage to the Mexican holiday Dia de los Muertos (Day of the Dead) and showcases the vibrant culture and music of Mexico.
\n - Research Trips to Mexico: The filmmakers conducted extensive research trips to Mexico to immerse themselves in the culture and traditions depicted in the film. They visited cities, villages, and cemeteries to capture the essence of Dia de los Muertos and gather inspiration for the story and visuals.
\n - Story Inspired by Personal Experiences: Director Lee Unkrich was inspired to create "Coco" after experiencing the joyous celebrations of Dia de los Muertos during his travels in Mexico. He wanted to create a film that honored the traditions and values of Mexican culture.
\n - Groundbreaking Animation: "Coco" features groundbreaking animation, particularly in its depiction of the Land of the Dead. The film's vibrant colors, intricate details, and lifelike character animations earned widespread praise from audiences and critics alike.
\n - Music and Original Songs: The film's soundtrack features a blend of traditional Mexican music and original songs composed by Germaine Franco, Adrian Molina, and others. The song "Remember Me," performed by various characters in the film, became a standout hit and won the Academy Award for Best Original Song.
\n - Voice Cast: The voice cast of "Coco" includes a mix of seasoned actors and newcomers, many of whom have Mexican heritage. Anthony Gonzalez voices the protagonist, Miguel, while Gael García Bernal, Benjamin Bratt, Alanna Ubach, and Edward James Olmos also lend their voices to the film.
\n - Awards and Accolades: "Coco" received widespread critical acclaim and won numerous awards, including the Academy Award for Best Animated Feature. It became the first Pixar film to win the Best Animated Feature Oscar since "Up" in 2009.
\n - Representation and Diversity: "Coco" is praised for its authentic representation of Mexican culture and its celebration of diversity. The film resonated with audiences around the world, sparking conversations about the importance of cultural representation in media.
\n - Emotional Impact: "Coco" is known for its emotional depth and heartfelt storytelling. The film explores themes of family, memory, and the power of music, leaving a lasting impression on viewers of all ages.
\n - Legacy and Cultural Impact: "Coco" has left a lasting legacy and continues to be celebrated as a cultural landmark. It has inspired merchandise, theme park attractions, and educational initiatives that celebrate Mexican culture and traditions.""")
if option == "Elemental":
    st.title("Elemental")
    st.image("https://preview.redd.it/361mhz9n11rb1.png?width=640&crop=smart&auto=webp&s=f8984f0e465cddbfe95f576bb2bad66c3bb1bb4f", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2769667097/?playlistId=tt15789038&ref_=ext_shr_lnk")
    st.write("Year: 2023")
    st.write("Duration: 1h41min")
    st.write("IMDB rating: 7.0")
    st.write("Argument: Follows Ember and Wade, in a city where fire, water, earth and air residents live together.")
    st.write("Directors: Peter Sohn")
    st.write("Writers: John Hoberg, Kat Likkel, Brenda Hsueh")
    st.write("Voices: Leah Lewis, Mamoudou Athie, Ronnie Del Carmen")
    st.write("""Fun Facts:
\n - Elemental's Creators Had to Upgrade the Cores and Buy More Computers: The Elemental People in Elemental proved to be more challenging than other past Pixar endeavors. The characters in Elemental all had such unique ways of interacting with their environment that creators had to buy more computers and upgrade them to achieve the characters' desired effect.
\n - Wade's Sibling's Girlfriend Is Named Ghibli: Lasseter's enthusiasm for Miyazaki's works helped bring Ghibli stories to the Western audience.
\n - Water People Were the Hardest to Animate: When it came to Water People, animators faced the challenge of making them act and behave like water.
\n - Wade Ripple Continuously Changes Colors: Every new scene calls for different lighting, which makes Wade change colors to reflect where he is.
\n - Director Peter Sohn Says He Would Be a Water Person: Like Wade, Sohn cries easily, so he believes water would best suit his personality.
\n - Elemental Is Pixar's First Romantic Comedy: Elemental is the first one to feature romantic love as a main plotline.
\n - Elemental's Animators Studied Water Balloon Movement
\n - Animators Watched POV City Tours Online: Elemental was going through development during the heaviest COVID lockdowns. To see how real cities worked, animators watched Point-of-View City Tours on YouTube.
\n - Elemental's Ending Almost Included a Steam Baby: an idea was floated around that the movie would end with Wade and Ember having a baby made of steam.
\n - Elemental Is Based on Peter Sohn's Life: like Ember, Sohn fell in love with someone with a different culture to his, which caused tension between his parents.""")
if option == "Good Dinosaur":
    st.title("Good Dinosaur")
    st.image("https://m.media-amazon.com/images/I/614ZCfw8+LL._AC_UF894,1000_QL80_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2104275737/?playlistId=tt1979388&ref_=ext_shr_lnk")
    st.write("Year: 2015")
    st.write("Duration: 1h33min")
    st.write("IMDB rating: 6.7")
    st.write("Argument: In a world where dinosaurs and humans live side-by-side, an Apatosaurus named Arlo makes an unlikely human friend.")
    st.write("Directors: Peter Sohn")
    st.write("Writers: Bob Peterson, Peter Sohn, Erik Benson")
    st.write("Voices: Jeffrey Wright, Frances McDormand, Maleah Nipay-Padilla")
    st.write("""Fun Facts:
\n - Alternate History Setting: "The Good Dinosaur" takes place in an alternate history where dinosaurs never became extinct. The film explores what might have happened if dinosaurs had continued to evolve and live alongside humans.
\n - Visual Realism: Pixar aimed for photorealistic visuals in "The Good Dinosaur," with detailed landscapes and realistic environments. The animation team used advanced techniques to create breathtaking scenery, including rivers, mountains, and forests.
\n - Unique Animation Style: The character designs in "The Good Dinosaur" feature a contrast between highly realistic environments and more stylized, cartoony characters. This juxtaposition creates a visually striking aesthetic that sets the film apart from other Pixar movies.
\n - Long Development Process: "The Good Dinosaur" had a notoriously long and troubled development process. Originally slated for release in 2014, the film underwent significant changes, including a complete overhaul of the story and character designs.
\n - Voice Cast: The film features an ensemble voice cast, including Raymond Ochoa as Arlo, Jack Bright as Spot, Sam Elliott as Butch, Anna Paquin as Ramsey, and Jeffrey Wright as Poppa. Each actor brings their character to life with memorable performances.
\n - Emotional Storytelling: Despite its whimsical premise, "The Good Dinosaur" delivers a heartfelt and emotional story about friendship, survival, and self-discovery. The bond between Arlo, the timid Apatosaurus, and Spot, the feral human boy, forms the emotional core of the film.
\n - T-Rex Ranch: One of the memorable sequences in the film takes place at T-Rex Ranch, where Arlo and Spot encounter a family of Tyrannosaurus Rexes. The T-Rexes are portrayed as ranchers, with Butch as the tough but compassionate leader.
\n - Visual Effects Innovation: Pixar's animation team developed new techniques for simulating natural phenomena in "The Good Dinosaur," including water, fire, and atmospheric effects. These innovations pushed the boundaries of computer-generated imagery in animated films.
\n - Scenic Inspirations: The landscapes in "The Good Dinosaur" were inspired by real-world locations, including the American Northwest and Wyoming's Jackson Hole region. The animators aimed to capture the awe-inspiring beauty of nature in their digital environments.
\n - Critical Reception: While "The Good Dinosaur" received mixed reviews from critics, it was praised for its stunning animation and heartfelt moments. Despite its troubled production history, the film resonated with audiences and remains a visually stunning addition to the Pixar canon.""")
if option == "Incredibles":
    st.title("Incredibles")
    st.image("https://popcultart.com/cdn/shop/products/The_Incredibles_-_Rico_Jr-min.jpg?v=1624891964&width=2048", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi3934427929/?ref_=ext_shr_lnk")
    st.write("Year: 2005")
    st.write("Duration: 1h55min")
    st.write("IMDB rating: 8.0")
    st.write("Argument: While trying to lead a quiet suburban life, a family of undercover superheroes are forced into action to save the world.")
    st.write("Directors: Brad Bird")
    st.write("Writers: Brad Bird")
    st.write("Voices: Craig T. Nelson, Samuel L. Jackson, Holly Hunter")
    st.write("""Fun Facts:
\n - Superhero Family Dynamic: "The Incredibles" is the first Pixar film to focus on human characters and delve into the superhero genre. It follows the Parr family, who must balance their secret identities as superheroes with their suburban family life.
\n - Brad Bird's Vision: Director Brad Bird drew inspiration from classic superhero comics and spy thrillers of the 1960s and 1970s for the film's aesthetic and tone. He envisioned "The Incredibles" as an homage to the golden age of superheroes.
\n - Elastigirl's Powers: Helen Parr, also known as Elastigirl, has the ability to stretch and contort her body into various shapes. Her elastic powers were inspired by Mr. Fantastic from Marvel Comics' "Fantastic Four" series.
\n - Edna Mode: The iconic character of Edna Mode, the eccentric fashion designer to the superheroes, was voiced by director Brad Bird himself. Her distinctive look and personality have made her a fan favorite.
\n - Score by Michael Giacchino: The film's score was composed by Michael Giacchino, who later became one of Pixar's go-to composers. Giacchino's dynamic and heroic score perfectly complements the action-packed sequences and emotional moments in the film.
\n - Family Dynamics: The Parr family's dynamic was inspired by Brad Bird's own experiences as a parent. He wanted to explore themes of family, identity, and the challenges of balancing personal and professional responsibilities.
\n - Critical and Commercial Success: "The Incredibles" was both a critical and commercial success, receiving widespread acclaim for its storytelling, animation, and action sequences. It won the Academy Award for Best Animated Feature in 2005.
\n - Sequel: Due to the film's popularity, a sequel titled "Incredibles 2" was released in 2018, directed once again by Brad Bird. The sequel picks up immediately after the events of the first film and explores the role reversal of Helen and Bob Parr.
\n - Syndrome's Design: The villain Syndrome's design was inspired by Brad Bird's childhood drawings of villains with large, square heads. His character embodies the classic comic book trope of the disgruntled fan turned villain.
\n - Cameo Appearances: Pixar's tradition of including Easter eggs and cameo appearances can be seen in "The Incredibles." For example, the Pizza Planet truck from "Toy Story" makes a brief appearance during the chase scene.""")
if option == "Inside Out":
    st.title("Inside Out")
    st.image("https://filmartgallery.com/cdn/shop/products/Inside-Out-Vintage-Movie-Poster-Original-1-Sheet-27x41_5000x.webp?v=1673039121", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi203730969/?playlistId=tt2096673&ref_=ext_shr_lnk")
    st.write("Year: 2015")
    st.write("Duration: 1h35min")
    st.write("IMDB rating: 8.1")
    st.write("Argument: After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions - Joy, Fear, Anger, Disgust and Sadness - conflict on how best to navigate a new city, house, and school.")
    st.write("Directors: Pete Docter, Ronnie Del Carmen")
    st.write("Writers: Pete Docter, Ronnie Del Carmen, Meg LeFauve")
    st.write("Voices: Amy Poehler, Bill Hader, Lewis Black")
    st.write("""Fun Facts:
\n - Emotionally Complex: "Inside Out" explores the inner workings of the mind and emotions, presenting a complex and nuanced portrayal of human psychology. The film delves into the emotions of joy, sadness, fear, anger, and disgust, personified as characters living inside the mind of an 11-year-old girl named Riley.
\n - Director's Personal Connection: Director Pete Docter was inspired to create "Inside Out" by his experiences as a parent, observing his daughter's emotional development. He wanted to explore the importance of embracing all emotions, including sadness.
\n - Emotion Consultants: To ensure authenticity in depicting emotions, the filmmakers consulted with psychologists and experts in the field of emotion research. They drew upon scientific research and personal experiences to create the characters and scenarios in the film.
\n - Star-Studded Voice Cast: "Inside Out" features a star-studded voice cast, including Amy Poehler as Joy, Phyllis Smith as Sadness, Bill Hader as Fear, Lewis Black as Anger, Mindy Kaling as Disgust, and Richard Kind as Bing Bong. Each actor brings their character to life with humor and heart.
\n - Colorful Animation: The film's animation features vibrant colors and imaginative designs, particularly in the abstract representation of Riley's mind. Each emotion has its own distinct color palette and visual style, reflecting their personalities.
\n - Award-Winning Soundtrack: The film's musical score was composed by Michael Giacchino, who also worked on other Pixar films like "Up" and "Ratatouille." Giacchino's score for "Inside Out" received critical acclaim and won the Academy Award for Best Original Score.
\n - Bing Bong's Character Design: Bing Bong, Riley's imaginary friend, was originally conceived as a monster-like creature. However, the character evolved into a more whimsical and lovable design inspired by vintage stuffed animals and carnival toys.
\n - Creative Storytelling: "Inside Out" employs creative storytelling techniques to explore complex psychological concepts in a way that is accessible to audiences of all ages. The film blends humor, adventure, and emotion to deliver a powerful and uplifting message.
\n - Impactful Themes: The film's themes of empathy, resilience, and the importance of embracing all emotions resonated with audiences worldwide. "Inside Out" sparked conversations about mental health and emotional well-being, particularly among children and parents.
\n - Critical and Commercial Success: "Inside Out" was both a critical and commercial success, earning over $858 million worldwide and receiving widespread acclaim from critics and audiences. It won the Academy Award for Best Animated Feature in 2016.""")
if option == "Luca":
    st.title("Luca")
    st.image("https://m.media-amazon.com/images/I/71kDOwRD78S._AC_UF894,1000_QL80_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi3001794585/?playlistId=tt12801262&ref_=ext_shr_lnk")
    st.write("Year: 2021")
    st.write("Duration: 1h35min")
    st.write("IMDB rating: 7.4")
    st.write("Argument: On the Italian Riviera, an unlikely but strong friendship grows between a human being and a sea monster disguised as a human.")
    st.write("Directors: Enrico Casarosa")
    st.write("Writers: Enrico Casarosa, Jesse Andrews, Simon Stephenson")
    st.write("Voices: Jacob Tremblay, Jack Dylan Grazer, Emma Berman")
    st.write("""Fun Facts:
\n - Italian Riviera Setting: "Luca" is set in a fictional seaside town on the Italian Riviera inspired by the picturesque coastal villages of the Liguria region in northwest Italy. The film beautifully captures the charm and warmth of Italian culture.
\n - Directorial Debut: "Luca" marks the feature directorial debut of Enrico Casarosa, who previously directed the Pixar short film "La Luna." Casarosa drew inspiration from his own childhood memories of summers spent by the sea in Italy.
\n - Friendship and Coming-of-Age: At its core, "Luca" is a heartwarming story about friendship, self-discovery, and coming-of-age. The film follows the adventures of Luca Paguro, a young sea monster, and his best friend Alberto Scorfano as they explore the human world above the surface.
\n - Unique Animation Style: The animation style of "Luca" is characterized by its vibrant colors, stylized character designs, and expressive visuals. The film's art direction draws inspiration from Italian art and culture, creating a visually stunning and immersive experience.
\n - Voice Cast: The voice cast of "Luca" features a mix of talented actors, including Jacob Tremblay as Luca, Jack Dylan Grazer as Alberto, Emma Berman as Giulia, Maya Rudolph as Luca's mother Daniela, and Jim Gaffigan as Luca's father Lorenzo.
\n - Inspirations from Miyazaki: Director Enrico Casarosa has cited the films of Japanese animator Hayao Miyazaki, particularly "Porco Rosso" and "Kiki's Delivery Service," as influences on the tone and storytelling of "Luca." Like Miyazaki's films, "Luca" captures the magic of childhood and the wonders of the natural world.
\n - Sea Monster Lore: In "Luca," sea monsters are depicted as shapeshifters who transform into human form when they emerge from the water. The film explores the mythical lore and legends surrounding sea monsters while putting a fresh spin on the classic concept.
\n - Italian Culture and Cuisine: "Luca" celebrates Italian culture and cuisine, showcasing mouthwatering dishes such as pasta, gelato, and seafood. The film invites viewers to savor the flavors and traditions of Italy through its rich visual storytelling.
\n - Homage to Childhood Adventures: The friendship between Luca and Alberto is reminiscent of the carefree adventures and discoveries of childhood. Their summer escapades, from riding Vespa scooters to participating in a local triathlon, evoke a sense of nostalgia and wonder.
\n - Universal Themes: Despite its specific Italian setting, "Luca" explores universal themes of acceptance, belonging, and the courage to be true to oneself. The film's message resonates with audiences of all ages and backgrounds, making it a timeless and relatable story.""")
if option == "Monsters Inc.":
    st.title("Monsters Inc.")
    st.image("https://image.tmdb.org/t/p/original/uG034IRI6lMd99WfzmrWSOARWZG.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi518890521/?playlistId=tt0198781&ref_=ext_shr_lnk")
    st.write("Year: 2001")
    st.write("Duration: 1h32min")
    st.write("IMDB rating: 8.1")
    st.write("Argument: In order to power the city, monsters have to scare children so that they scream. However, the children are toxic to the monsters, and after a child gets through, two monsters realize things may not be what they think.")
    st.write("Directors: Pete Docter, David Silverman, Lee Unkrich")
    st.write("Writers: Enrico Casarosa, Jesse Andrews, Simon Stephenson")
    st.write("Voices: Billy Crystal, John Goodman, Mary Gibbs")
    st.write("""Fun Facts:
\n - Unique Concept: "Monsters, Inc." explores the idea that monsters in children's closets are actually employees of a corporation that harvests screams as a source of energy. It's a creative twist on the classic childhood fear of monsters under the bed.
\n - Directorial Debut: "Monsters, Inc." marked the directorial debut of Pete Docter, who later went on to direct other Pixar classics like "Up" and "Inside Out." Docter's imaginative storytelling and sense of humor are evident throughout the film.
\n - Dynamic Duo: The film follows the adventures of two main characters: Sulley (voiced by John Goodman), a big, blue, furry monster, and his best friend Mike Wazowski (voiced by Billy Crystal), a small, green, one-eyed monster. Their friendship forms the heart of the story.
\n - Monstropolis: The film is set in the bustling city of Monstropolis, where monsters live and work. The city's unique design blends elements of urban life with whimsical monster-inspired architecture and technology.
\n - Scare Floor: The central location in "Monsters, Inc." is the scare floor of the Monsters, Inc. factory, where monsters clock in to scare children and collect screams. The factory's elaborate scare floor design and conveyor belt system add to the film's visual appeal.
\n - Boo: One of the most beloved characters in "Monsters, Inc." is Boo, a young human girl who accidentally enters the monster world. Boo's innocence and curiosity provide both comedic moments and emotional depth to the story.
\n - Randall Boggs: The film's antagonist, Randall Boggs (voiced by Steve Buscemi), is a chameleon-like monster who schemes to increase his scare output by any means necessary. His villainous antics provide plenty of suspense and excitement.
\n - Laugh Factory: In the latter part of the film, the monsters discover that laughter is a more powerful source of energy than screams. This realization leads to the creation of the laugh factory, where monsters are trained to elicit laughter from children instead of scares.
\n - Visual Effects: "Monsters, Inc." showcased groundbreaking animation and visual effects for its time, particularly in the depiction of fur and texture. The animators faced technical challenges in creating believable fur for Sulley and other monsters but ultimately achieved stunning results.
\n - Legacy: "Monsters, Inc." became a beloved classic and spawned a prequel film, "Monsters University," which explores the origins of Sulley and Mike's friendship during their college years. The original film's memorable characters, humor, and heartwarming story continue to enchant audiences of all ages.""")
if option == "Finding Nemo":
    st.title("Finding Nemo")
    st.image("https://filmartgallery.com/cdn/shop/files/Finding-Nemo-Vintage-Movie-Poster-Original-1-Sheet-27x41_5000x.jpg?v=1684357269", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2687214105/?playlistId=tt0266543&ref_=ext_shr_lnk")
    st.write("Year: 2003")
    st.write("Duration: 1h40min")
    st.write("IMDB rating: 8.2")
    st.write("Argument: After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.")
    st.write("Directors: Andrew Stanton, Lee Unkrich")
    st.write("Writers: Andrew Stanton, Bob Peterson, David Reynolds")
    st.write("Voices: Albert Brooks, Ellen DeGeneres, Alexander Gould")
    st.write("""Fun Facts:
\n - Inspiration from a Documentary: The idea for "Finding Nemo" was sparked by a documentary about fish and coral reefs that director Andrew Stanton watched. The film's stunning underwater visuals and diverse marine life were influenced by real-life marine ecosystems.
\n - Character Development: The character of Marlin, a clownfish and the film's protagonist, was inspired by director Andrew Stanton's own experience as a father. Marlin's journey to find his son, Nemo, reflects themes of parenthood, love, and overcoming adversity.
\n - Voice Cast: The film features an all-star voice cast, including Albert Brooks as Marlin, Ellen DeGeneres as Dory, Alexander Gould as Nemo, Willem Dafoe as Gill, and Brad Garrett as Bloat. Each actor brings their character to life with humor and heart.
\n - Ellen DeGeneres' Memorable Role: Ellen DeGeneres' portrayal of Dory, a forgetful blue tang fish, became one of the most iconic and beloved characters in Pixar's history. Dory's optimistic attitude and humorous antics stole the hearts of audiences worldwide.
\n - Underwater Adventure: "Finding Nemo" takes audiences on an unforgettable underwater adventure through the Great Barrier Reef and beyond. The film's stunning animation and detailed underwater environments immerse viewers in the colorful world beneath the sea.
\n - Advanced Animation Techniques: Pixar's animators developed innovative techniques to simulate underwater physics and movement in "Finding Nemo." They used computer algorithms to create realistic water effects and animate the fluid motion of fish and sea creatures.
\n - Environmental Themes: The film raises awareness about environmental conservation and the importance of protecting marine ecosystems. Through its depiction of pollution, habitat destruction, and overfishing, "Finding Nemo" highlights the impact of human activities on marine life.
\n - Box Office Success: "Finding Nemo" was a massive box office success, grossing over $940 million worldwide and becoming the highest-grossing animated film at the time of its release. It received widespread critical acclaim and won the Academy Award for Best Animated Feature.
\n - Spin-Off and Sequel: The success of "Finding Nemo" led to a spin-off film titled "Finding Dory," which focuses on Dory's quest to reunite with her long-lost family. The film was released in 2016 and became another box office hit for Pixar.
\n - Cultural Impact: "Finding Nemo" has had a lasting cultural impact, inspiring merchandise, theme park attractions, and even marine conservation initiatives. The film's endearing characters, memorable quotes, and timeless message of friendship continue to resonate with audiences of all ages.""")
if option == "Onward":
    st.title("Onward")
    st.image("https://m.media-amazon.com/images/I/71YwxjfhEiL._AC_UF894,1000_QL80_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi1287503385/?playlistId=tt7146812&ref_=ext_shr_lnk")
    st.write("Year: 2020")
    st.write("Duration: 1h42min")
    st.write("IMDB rating: 7.4")
    st.write("Argument: In a magical world full of technological advances, elven brothers Ian and Barley Lightfoot set out on an adventure to resurrect their late father for a day.")
    st.write("Directors: Dan Scalon")
    st.write("Writers: Dan Scanlon, Keith Bunin, Jason Headley")
    st.write("Voices: Tom Holland, Chris Pratt, Julia Louis-Dreyfus")
    st.write("""Fun Facts:
\n - Urban Fantasy Setting: "Onward" takes place in a suburban fantasy world where mythical creatures like elves, trolls, and unicorns live in a modern-day setting. The film blends elements of fantasy with everyday life, creating a unique and imaginative world.
\n - Brotherly Bond: At its heart, "Onward" is a story about brotherhood and the bond between siblings. The film follows Ian and Barley Lightfoot, two elf brothers on a quest to bring back their late father for one magical day.
\n - Directorial Debut: "Onward" marked the directorial debut of Dan Scanlon, who previously worked as a story artist on Pixar's "Cars" and "Toy Story 3." Scanlon drew inspiration from his own experiences growing up without a father and his relationship with his brother.
\n - Voice Cast: The film features a talented voice cast, including Tom Holland as Ian Lightfoot, Chris Pratt as Barley Lightfoot, Julia Louis-Dreyfus as Laurel Lightfoot, and Octavia Spencer as the Manticore. The actors' performances bring depth and emotion to the characters.
\n - Modern Fantasy Elements: "Onward" incorporates modern fantasy elements such as smartphones, cars, and other modern conveniences into its magical world. This blending of fantasy and contemporary technology adds humor and relatability to the story.
\n - Quest Narrative: The central plot of "Onward" follows Ian and Barley as they embark on a quest to find a magical artifact that will allow them to bring back their father for a day. Along the way, they encounter mythical creatures, obstacles, and challenges.
\n - Subversion of Tropes: "Onward" subverts traditional fantasy tropes by focusing on characters who are not typical heroes or adventurers. Ian and Barley are portrayed as ordinary teenagers who must discover their inner strength and courage to complete their quest.
\n - Inspirations from Role-Playing Games: The film draws inspiration from classic role-playing games (RPGs) like Dungeons & Dragons, with characters going on quests, battling monsters, and casting spells. This RPG-inspired storytelling adds depth and excitement to the adventure.
\n - Emotional Themes: "Onward" explores themes of loss, grief, and the power of familial love. The film's heartfelt message about cherishing the time we have with loved ones resonates with audiences of all ages.
\n - Visual Spectacle: Like all Pixar films, "Onward" features stunning animation and visual effects. The film's magical creatures, spellcasting, and fantastical landscapes are brought to life with vibrant colors and intricate detail.""")
if option == "Ratatouille":
    st.title("Ratatouille")
    st.image("https://cdn.europosters.eu/image/1300/posters/ratatouille-una-hoja-i6730.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi906147865/?playlistId=tt0382932&ref_=ext_shr_lnk")
    st.write("Year: 2007")
    st.write("Duration: 1h51min")
    st.write("IMDB rating: 8.1")
    st.write("Argument: A rat who can cook makes an unusual alliance with a young kitchen worker at a famous Paris restaurant.")
    st.write("Directors: Brad Bird, Jan Pinkava")
    st.write("Writers: Brad Bird, Jan Pinkava, Jim Capobianco")
    st.write("Voices: Brad Garrett, Lou Romano, Patton Oswalt")
    st.write("""Fun Facts:
\n - Unique Concept: "Ratatouille" is unique among Pixar films for its premise, which revolves around a rat named Remy who dreams of becoming a chef in Paris. The film explores themes of ambition, passion, and the pursuit of dreams.
\n - Director's Passion Project: Director Brad Bird was passionate about the idea of a rat becoming a chef and pitched the concept to Pixar. Despite initial skepticism, Bird's vision for the film won over the studio, and "Ratatouille" went into production.
\n - Research Trip to Paris: To capture the essence of Parisian cuisine and culture, the filmmakers embarked on a research trip to Paris. They visited renowned restaurants, bakeries, and markets to immerse themselves in the culinary world depicted in the film.
\n - Gusteau's Restaurant: The fictional restaurant featured in "Ratatouille," Gusteau's, is named after the renowned chef Auguste Gusteau. Gusteau serves as a mentor and inspiration to Remy, and his motto, "Anyone can cook," becomes a central theme of the film.
\n - Voice Cast: The film features a talented voice cast, including Patton Oswalt as Remy, Lou Romano as Linguini, Brad Garrett as Gusteau, Janeane Garofalo as Colette, and Peter O'Toole as Anton Ego. Each actor brings depth and personality to their respective characters.
\n - Culinary Accuracy: Pixar went to great lengths to ensure culinary accuracy in "Ratatouille." They consulted with professional chefs, including Thomas Keller and Guy Savoy, to accurately depict kitchen techniques, food preparation, and fine dining etiquette.
\n - Gourmet Animation: The animation team paid meticulous attention to detail when animating food in "Ratatouille." Every dish, ingredient, and cooking utensil was meticulously crafted to look as realistic and appetizing as possible.
\n - Remy's Ratatouille: The dish "ratatouille" featured in the film is a traditional French Provençal stewed vegetable dish. Remy's version of ratatouille, which he prepares for the critic Anton Ego, is presented in a visually stunning and artistically composed manner.
\n - Critical Acclaim: "Ratatouille" received widespread critical acclaim upon its release, with praise for its animation, storytelling, and themes. The film won the Academy Award for Best Animated Feature and was nominated for four other Oscars.
\n - Legacy: "Ratatouille" has left a lasting legacy and continues to be celebrated as one of Pixar's most beloved films. It has inspired merchandise, theme park attractions, and even a real-life restaurant based on Gusteau's called "Bistrot Chez Rémy" at Disneyland Paris.""")
if option == "Soul":
    st.title("Soul")
    st.image("https://filmspot.com.pt/images/filmes/posters/big/508442_pt.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi1257423129/?playlistId=tt2948372&ref_=ext_shr_lnk")
    st.write("Year: 2020")
    st.write("Duration: 1h40min")
    st.write("IMDB rating: 8.0")
    st.write("Argument: Joe is a middle-school band teacher whose life hasn't quite gone the way he expected. His true passion is jazz. But when he travels to another realm to help someone find their passion, he soon discovers what it means to have soul.")
    st.write("Directors: Pete Docter, Kemp Powers")
    st.write("Writers: Pete Docter, Mike Jones, Kemp Powers")
    st.write("Voices: Jamie Foxx, Tina Fey, Graham Norton")
    st.write("""Fun Facts:
\n - Groundbreaking Animation: "Toy Story" was the first feature-length computer-animated film ever created. Its release in 1995 marked a significant milestone in the history of animation, revolutionizing the industry with its cutting-edge technology.
\n - Collaborative Development: The development of "Toy Story" involved collaboration between Pixar Animation Studios and Disney. Pixar's creative team, led by John Lasseter, worked closely with Disney's animators and executives to bring the story to life.
\n - Character Designs: The characters in "Toy Story" were designed to be appealing and relatable to audiences of all ages. The design team drew inspiration from classic toys and childhood favorites, creating iconic characters like Woody, Buzz Lightyear, and Mr. Potato Head.
\n - Voice Cast: "Toy Story" features a star-studded voice cast, including Tom Hanks as Woody, Tim Allen as Buzz Lightyear, Don Rickles as Mr. Potato Head, Jim Varney as Slinky Dog, and Wallace Shawn as Rex. The actors' performances bring depth and humor to their characters.
\n - Innovative Storytelling: "Toy Story" pioneered a new form of storytelling in animation, blending humor, heart, and adventure to appeal to audiences of all ages. The film's relatable themes of friendship, loyalty, and self-discovery resonated with viewers worldwide.
\n - Cultural Impact: "Toy Story" became a cultural phenomenon upon its release, captivating audiences with its groundbreaking animation and lovable characters. It spawned a franchise that includes sequels, spin-offs, merchandise, and theme park attractions.
\n - Pixar's Legacy: "Toy Story" solidified Pixar's reputation as a leading animation studio and paved the way for future successes. The film's critical and commercial success established Pixar as a powerhouse in the animation industry.
\n - Technical Achievements: The animation team faced numerous technical challenges in creating "Toy Story," including rendering realistic textures, lighting, and character movements. They developed innovative techniques and software to overcome these obstacles and achieve stunning visual effects.
\n - Nostalgic References: "Toy Story" is filled with nostalgic references to classic toys and pop culture, appealing to both children and adults. The film pays homage to beloved toys from different eras, from cowboy dolls to space action figures.
\n - Enduring Legacy: Over 25 years after its release, "Toy Story" remains a beloved classic and a cultural touchstone for multiple generations. Its timeless themes, memorable characters, and groundbreaking animation continue to captivate audiences around the world.""")
if option == "Toy Story":
    st.title("Toy Story")
    st.image("https://m.media-amazon.com/images/I/71iSIVGZQUL._AC_UF1000,1000_QL80_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2052129305/?playlistId=tt0114709&ref_=ext_shr_lnk")
    st.write("Year: 1995")
    st.write("Duration: 1h21min")
    st.write("IMDB rating: 8.3")
    st.write("Argument: A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom.")
    st.write("Directors: John Lasseter")
    st.write("Writers: John Lasseter, Pete Docter, Andrew Stanton")
    st.write("Voices: Tom Hanks, Tim Allen, Don Rickles")
    st.write("""Fun Facts:
\n - First Feature-Length Computer-Animated Film: "Toy Story" made history as the first feature-length film to be entirely computer-animated. Released in 1995, it revolutionized the animation industry and set a new standard for animated filmmaking.
\n - Originally a Buddy Comedy: In its early stages of development, "Toy Story" was conceived as a buddy comedy between a ventriloquist's dummy and a tin toy. However, the story evolved over time to focus on the relationship between Woody and Buzz Lightyear.
\n - Inspiration from Lasseter's Childhood Toys: Director John Lasseter drew inspiration from his own childhood toys, particularly a pull-string talking doll and a space action figure. These toys served as the basis for the characters of Woody and Buzz Lightyear.
\n - Extensive Voice Cast Auditions: Before casting Tom Hanks as the voice of Woody and Tim Allen as Buzz Lightyear, Pixar conducted extensive auditions with various actors. Hanks was initially hesitant to take on the role, but he was won over by the script and Pixar's innovative approach to animation.
\n - Hidden Easter Eggs: "Toy Story" is known for its hidden Easter eggs and references to other Pixar films. For example, the carpet in Sid's house features the same pattern as the one in the Overlook Hotel from Stanley Kubrick's "The Shining."
\n - Cameo Appearances: Several iconic toys make cameo appearances in "Toy Story," including Mr. Potato Head, Etch A Sketch, and Barrel of Monkeys. These toys add authenticity to the film's toy-filled world and evoke nostalgia for viewers.
\n - Groundbreaking Animation Techniques: Pixar's animators developed groundbreaking animation techniques to bring the characters to life in "Toy Story." They focused on creating realistic movements and expressions for the toys, making them feel alive and relatable to audiences.
\n - Pixar's First Franchise: "Toy Story" launched Pixar's first franchise, spawning three sequels, multiple spin-off shorts, and a variety of merchandise. The success of the franchise cemented Pixar's reputation as a leading animation studio.
\n - Critical and Commercial Success: "Toy Story" was a critical and commercial success, grossing over $373 million worldwide and receiving widespread acclaim from critics and audiences alike. It was nominated for three Academy Awards, including Best Original Screenplay.
\n - Cultural Impact: "Toy Story" has had a significant cultural impact since its release, influencing popular culture and inspiring a generation of filmmakers and animators. Its themes of friendship, loyalty, and self-discovery continue to resonate with audiences of all ages.""")
if option == "Turning Red":
    st.title("Turning Red")
    st.image("https://m.media-amazon.com/images/M/MV5BOWYxZDMxYWUtNjNiZC00MDE1LWI2Y2QtNWZhNDAyMGY5ZjVhXkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi3974546201/?playlistId=tt8097030&ref_=ext_shr_lnk")
    st.write("Year: 2022")
    st.write("Duration: 1h40min")
    st.write("IMDB rating: 7.0")
    st.write("Argument: A thirteen-year-old girl named Mei Lee is torn between staying her mother's dutiful daughter and the changes of adolescence. And as if the challenges were not enough, whenever she gets overly excited she transforms into a giant red panda.")
    st.write("Directors: Domee Shi")
    st.write("Writers: John Lasseter, Pete Docter, Andrew Stanton")
    st.write("Voices: Domee Shi, Julia Cho, Sarah Streicher")
    st.write("""Fun Facts:
\n - First Pixar Film Directed by a Woman: "Turning Red" is directed by Domee Shi, making her the first woman to direct a feature film for Pixar Animation Studios. Shi previously won an Academy Award for her animated short film "Bao."
\n - Inspired by Shi's Childhood: The story of "Turning Red" is inspired by Domee Shi's own experiences growing up as a Chinese-Canadian girl in Toronto. The film explores themes of identity, family, and adolescence through a unique cultural lens.
\n - Magical Realism: "Turning Red" combines elements of magical realism with coming-of-age storytelling. The protagonist, Mei Lee, transforms into a giant red panda whenever she experiences intense emotions, adding a fantastical twist to her teenage journey.
\n - Vibrant Toronto Setting: The film is set in Toronto, Canada, and showcases the city's diverse neighborhoods, landmarks, and cultural influences. Domee Shi drew upon her memories of growing up in Toronto to create an authentic and vibrant backdrop for the story.
\n - Voice Cast: "Turning Red" features an ensemble voice cast, including Rosalie Chiang as Mei Lee, Sandra Oh as Mei's mother Ming, and James Hong as Mei's grandfather. The cast brings depth and authenticity to their characters, infusing the film with humor and heart.
\n - Unique Animation Style: Pixar's animation team employed a unique animation style for "Turning Red" to capture the energy and spirit of the characters. The film's visual aesthetic blends traditional 2D animation with modern CGI techniques, resulting in a dynamic and colorful look.
\n - Music and Soundtrack: The film's soundtrack features a mix of contemporary pop music and original score compositions. The music reflects the film's themes of self-expression and individuality, adding emotional depth to key moments in the story.
\n - Cultural Representation: "Turning Red" celebrates Chinese culture and heritage through its characters, settings, and storytelling. Domee Shi drew inspiration from her own Chinese-Canadian background to infuse the film with authentic cultural details and traditions.
\n - Empowering Message: At its core, "Turning Red" delivers an empowering message about self-acceptance, embracing one's identity, and finding the courage to be true to oneself. The film encourages audiences to celebrate their uniqueness and stand tall in the face of societal expectations.
\n - Anticipated Release: "Turning Red" is highly anticipated by audiences around the world and is expected to be a groundbreaking addition to Pixar's roster of animated films. The film's diverse representation, heartfelt story, and innovative animation make it a must-watch for audiences of all ages.""")
if option == "Up":
    st.title("Up")
    st.image("https://i.ebayimg.com/images/g/9UoAAOSwp-1i7Hay/s-l1200.webp", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi543228697/?ref_=ext_shr_lnk")
    st.write("Year: 2009")
    st.write("Duration: 1h36min")
    st.write("IMDB rating: 8.3")
    st.write("Argument: 78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway.")
    st.write("Directors: Pete Docter, Bob Peterson")
    st.write("Writers: Pete Docter, Bob Peterson, Tom McCarthy")
    st.write("Voices: Edward Asner, Jordan Nagai, John Ratzenberger")
    st.write("""Fun Facts:
\n - Emotional Opening Sequence: "Up" opens with a poignant and wordless montage depicting the life of Carl and Ellie Fredricksen from childhood to old age. This emotionally resonant sequence, which spans decades of their lives, sets the tone for the rest of the film.
\n - Carl's Floating House: The iconic image of Carl Fredricksen's house being lifted into the air by thousands of balloons has become one of the most memorable scenes in Pixar history. The concept of a flying house was inspired by director Pete Docter's childhood fantasies of escaping the mundane world.
\n - Exploring Adult Themes: "Up" is notable for its exploration of adult themes such as grief, loss, and loneliness. The character of Carl Fredricksen, an elderly widower grappling with the loss of his wife, adds depth and emotional resonance to the story.
\n - Russell's Wilderness Explorer Badges: Russell, the young Wilderness Explorer who accompanies Carl on his adventure, wears a sash adorned with various merit badges. These badges were inspired by real-life scouting organizations and include humorous designs such as the "Assisting the Elderly" badge.
\n - Talking Dogs: One of the memorable characters in "Up" is Dug, a lovable golden retriever who can speak thanks to a special collar. Dug's ability to communicate with humans adds humor and charm to the film, leading to memorable lines like "Squirrel!"
\n - Paradise Falls: The fictional location of Paradise Falls, where Carl and Ellie dream of traveling to, was inspired by real-life tepui mountains in South America. The stunning vistas and rugged terrain of Paradise Falls serve as the backdrop for much of the film's adventure.
\n - Academy Award-Winning Score: Composer Michael Giacchino won an Academy Award for Best Original Score for his work on "Up." The film's uplifting and emotionally resonant score perfectly complements the storytelling and adds depth to the characters' journey.
\n - Kevin the Bird: Kevin, a colorful and exotic bird encountered by Carl and Russell in the jungle, was inspired by real-life flightless birds such as the cassowary and the emu. Kevin's quirky personality and expressive animation make her a memorable addition to the cast.
\n - Carl and Ellie's Adventure Book: Throughout the film, Carl carries a scrapbook filled with mementos and memories of his and Ellie's life together. The adventure book symbolizes their shared dreams and serves as a poignant reminder of their enduring love.
\n - Critical and Commercial Success: "Up" was a critical and commercial success upon its release, grossing over $735 million worldwide and receiving widespread acclaim from audiences and critics alike. It won two Academy Awards, including Best Animated Feature, and was nominated for Best Picture, making it the second animated film to receive this honor after "Beauty and the Beast.""")
if option == "Wall E":
    st.title("Wall E")
    st.image("https://static1.cbrimages.com/wordpress/wp-content/uploads/2020/03/wall-e-poster-pixar-june-27.jpg", width=200)
    st.link_button("Trailer", "https://www.imdb.com/video/vi2192703769/?playlistId=tt0910970&ref_=ext_shr_lnk")
    st.write("Year: 2008")
    st.write("Duration: 1h38min")
    st.write("IMDB rating: 8.4")
    st.write("Argument: In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind.")
    st.write("Directors: Andrew Stanton")
    st.write("Writers: Andrew Stanton, Pete Docter, Jim Reardon")
    st.write("Voices: Ben Burtt, Elissa Knight, Jeff Garlin")
    st.write("""Fun Facts:
\n - Minimal Dialogue: "WALL-E" is known for its minimal dialogue, with the majority of the film's first act featuring little to no spoken words. Instead, the story is told through visual storytelling and expressive animation, making it accessible to audiences of all languages.
\n - Environmental Message: "WALL-E" delivers a powerful environmental message about the consequences of consumerism, waste, and the importance of environmental stewardship. The film's depiction of a dystopian future Earth covered in garbage serves as a cautionary tale about the impact of human activities on the planet.
\n - Silent Film Influences: Director Andrew Stanton drew inspiration from silent films and classic Hollywood cinema when crafting the visual style and storytelling of "WALL-E." The film's charming character animation and expressive pantomime pay homage to the golden age of cinema.
\n - Robot Love Story: At its heart, "WALL-E" is a love story between two robots: WALL-E, a lonely waste-collecting robot, and EVE, a sleek and sophisticated probe sent to Earth on a mission. Their budding romance and mutual affection drive the emotional core of the film.
\n - Innovative Sound Design: Sound designer Ben Burtt created the distinctive sounds of WALL-E and EVE using a combination of mechanical noises, synthesized tones, and human vocalizations. These sounds help to convey the personalities and emotions of the characters without traditional dialogue.
\n - Awards and Accolades: "WALL-E" received widespread critical acclaim upon its release and won the Academy Award for Best Animated Feature. It was also nominated for Best Original Screenplay, making it the first animated film to be nominated in this category since "Toy Story" in 1995.
\n - Robot Cameos: "WALL-E" features cameo appearances by several iconic robots from popular culture, including R2-D2 and C-3PO from "Star Wars," as well as the T-800 from "The Terminator." These nods to other sci-fi franchises add a playful element to the film.
\n - Realistic Animation: Pixar's animators pushed the boundaries of computer animation with "WALL-E," creating realistic textures, lighting, and environments to bring the world of the film to life. The attention to detail in the animation adds depth and immersion to the storytelling.
\n - Human Characters: In contrast to the robotic protagonists, the human characters in "WALL-E" are portrayed as overweight, sedentary consumers who have become reliant on technology and convenience. The portrayal of humanity's future evolution adds a satirical element to the film's social commentary.
\n - Timeless Themes: Despite being set in a distant future, "WALL-E" explores timeless themes of love, friendship, resilience, and hope. The film's optimistic message about the power of love and the possibility of redemption resonates with audiences of all ages.""")