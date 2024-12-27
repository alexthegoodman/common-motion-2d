// use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};
// use derive_new::new;

// #[derive(new, Clone, Debug)]
// pub struct TextGenerationItem {
//     pub text: String,
// }

// #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
// pub struct DbPediaItem {
//     pub content: String,
// }

// pub struct DbPediaDataset {
//     dataset: SqliteDataset<DbPediaItem>,
// }

// impl Dataset<TextGenerationItem> for DbPediaDataset {
//     fn get(&self, index: usize) -> Option<TextGenerationItem> {
//         self.dataset
//             .get(index)
//             .map(|item| TextGenerationItem::new(item.content))
//     }

//     fn len(&self) -> usize {
//         self.dataset.len()
//     }
// }

// impl DbPediaDataset {
//     pub fn train() -> Self {
//         Self::new("train")
//     }

//     pub fn test() -> Self {
//         Self::new("test")
//     }
//     pub fn new(split: &str) -> Self {
//         let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
//             .dataset(split)
//             .unwrap();
//         Self { dataset }
//     }
// }

use burn::data::dataset::{Dataset, InMemDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

// #[derive(Deserialize, Serialize, Debug, Clone)]
// pub struct MotionPaths {
//     #[serde(rename = "POLYGON_INDEX")]
//     pub polygon_index: i32,

//     #[serde(rename = "TIME")]
//     pub time: f32,

//     #[serde(rename = "WIDTH")]
//     pub width: i32,

//     #[serde(rename = "HEIGHT")]
//     pub height: i32,

//     #[serde(rename = "X")]
//     pub x: i32,

//     #[serde(rename = "Y")]
//     pub y: i32,
// }

// pub struct MotionDataset {
//     dataset: InMemDataset<MotionPaths>,
// }

// impl MotionDataset {
//     pub fn new() -> Result<Self, std::io::Error> {
//         // Download dataset csv file
//         let path = MotionDataset::download_train();

//         let mut rdr = csv::ReaderBuilder::new();
//         // let rdr = rdr.delimiter(b'\t'); // we can use default , deliminator

//         let dataset = InMemDataset::from_csv(path, &rdr).unwrap();

//         let dataset = Self { dataset };

//         Ok(dataset)
//     }

//     pub fn train() -> Result<Self, std::io::Error> {
//         // Download dataset csv file
//         let path = MotionDataset::download_train();

//         let mut rdr = csv::ReaderBuilder::new();
//         let rdr = rdr.has_headers(false);
//         // let rdr = rdr.delimiter(b'\t'); // we can use default , deliminator

//         let dataset = InMemDataset::from_csv(path, &rdr).unwrap();

//         let dataset = Self { dataset };

//         Ok(dataset)
//     }

//     pub fn test() -> Result<Self, std::io::Error> {
//         // Download dataset csv file
//         let path = MotionDataset::download_test();

//         let mut rdr = csv::ReaderBuilder::new();
//         let rdr = rdr.has_headers(false);
//         // let rdr = rdr.delimiter(b'\t'); // we can use default , deliminator

//         let dataset = InMemDataset::from_csv(path, &rdr).unwrap();

//         let dataset = Self { dataset };

//         Ok(dataset)
//     }

//     fn download_train() -> PathBuf {
//         // Point file to current example directory
//         let backup_dir = Path::new("backup");
//         let file_name = backup_dir.join("train.csv");

//         if file_name.exists() {
//             println!("File already downloaded at {:?}", file_name);
//         };

//         file_name
//     }

//     fn download_test() -> PathBuf {
//         // Point file to current example directory
//         let backup_dir = Path::new("backup");
//         let file_name = backup_dir.join("test.csv");

//         if file_name.exists() {
//             println!("File already downloaded at {:?}", file_name);
//         };

//         file_name
//     }
// }

use std::io::{self, BufRead, BufReader};

pub struct MotionDataset {
    dataset: InMemDataset<String>,
}

impl MotionDataset {
    pub fn new() -> Result<Self, io::Error> {
        Self::train()
    }

    pub fn train() -> Result<Self, io::Error> {
        let path = MotionDataset::download_train();
        let dataset = Self::load_txt_dataset(&path)?;
        Ok(Self { dataset })
    }

    pub fn test() -> Result<Self, io::Error> {
        let path = MotionDataset::download_test();
        let dataset = Self::load_txt_dataset(&path)?;
        Ok(Self { dataset })
    }

    fn load_txt_dataset(path: &Path) -> Result<InMemDataset<String>, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut sequences = Vec::new();
        let mut current_sequence = String::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim() == "---" {
                if !current_sequence.is_empty() {
                    sequences.push(current_sequence);
                    current_sequence = String::new();
                }
            } else if !line.trim().is_empty() {
                current_sequence.push_str(&line);
                current_sequence.push(','); // comma delimiter to separate lines clearly
                current_sequence.push('\n');
            }
        }

        if !current_sequence.is_empty() {
            sequences.push(current_sequence);
        }

        Ok(InMemDataset::new(sequences))
    }

    fn download_train() -> PathBuf {
        let backup_dir = Path::new("backup");
        let file_name = backup_dir.join("train.txt");

        if file_name.exists() {
            println!("File already downloaded at {:?}", file_name);
        }

        file_name
    }

    fn download_test() -> PathBuf {
        let backup_dir = Path::new("backup");
        let file_name = backup_dir.join("test.txt");

        if file_name.exists() {
            println!("File already downloaded at {:?}", file_name);
        }

        file_name
    }
}

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

impl Dataset<TextGenerationItem> for MotionDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            // .map(|item| TextGenerationItem::new(item.content)) // content is usually a whole text block
            .map(|item| {
                // option 1 worst
                // let text = format!(
                //     "Polygon index: {},{},{},{},{},{}",
                //     item.polygon_index, item.time, item.width, item.height, item.x, item.y
                // );

                // // option 2 better
                // let text = format!(
                //     "{} {} {} {} {} {}",
                //     item.polygon_index, item.time, item.width, item.height, item.x, item.y
                // );

                // option 3 best?
                // The polygon index is 1, the time is 2, the width is 3, the height is 4, the x-coordinate is 5, and the y-coordinate is 6.

                // option 4
                // let text = format!(
                //     "{} {} {} {} {} {}",
                //     item.polygon_index, item.time, item.width, item.height, item.x, item.y
                // );

                // when pulling line by line
                // let window_size = 5;
                // let start_idx = index.saturating_sub(window_size / 2);
                // let end_idx = (index + window_size / 2).min(self.dataset.len());

                // let mut window_texts = Vec::new();

                // for i in start_idx..end_idx {
                //     if let Some(item) = self.dataset.get(i) {
                //         window_texts.push(format!(
                //             "{} {} {} {} {} {} ", // space at end could be important for parsing
                //             item.polygon_index, item.time, item.width, item.height, item.x, item.y
                //         ));
                //     }
                // }

                // let text = window_texts.join("\n");

                TextGenerationItem::new(item)
            })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
