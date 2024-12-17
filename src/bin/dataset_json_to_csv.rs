use csv::{Writer, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::from_str;
use std::fs::File;
use std::io;
use std::time::Duration;
use std::{fs, path::PathBuf};

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedState {
    pub sequences: Vec<Sequence>,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Sequence {
    pub id: String,
    pub active_polygons: Vec<SavedPolygonConfig>, // used for dimensions, etc
    pub polygon_motion_paths: Vec<AnimationData>,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct AnimationData {
    pub id: String,
    /// id of the associated polygon
    pub polygon_id: String,
    /// Total duration of the animation
    pub duration: Duration,
    /// Hierarchical property structure for UI
    pub properties: Vec<AnimationProperty>,
}

/// Represents a property that can be animated in the UI
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct AnimationProperty {
    /// Name of the property (e.g., "Position.X", "Rotation.Z")
    pub name: String,
    /// Path to this property in the data (for linking to MotionPath data)
    pub property_path: String,
    /// Nested properties (if any)
    pub children: Vec<AnimationProperty>,
    /// Direct keyframes for this property
    pub keyframes: Vec<UIKeyframe>,
    /// Visual depth in the property tree
    pub depth: u32,
}

/// Types of easing functions available for interpolation
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum EasingType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

/// Represents a keyframe in the UI
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct UIKeyframe {
    /// Used to associate with this speciifc UI Keyframe
    pub id: String,
    /// Time of the keyframe
    pub time: Duration,
    /// Value at this keyframe (could be position, rotation, etc)
    pub value: KeyframeValue,
    /// Type of interpolation to next keyframe
    pub easing: EasingType,
}

/// Possible values for keyframes
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum KeyframeValue {
    Position([i32; 2]),
    Rotation(i32),
    Scale(i32), // this will be 100 for default size to work with i32 and Eq
    PerspectiveX(i32),
    PerspectiveY(i32),
    Opacity(i32), // also out of 100
    Custom(Vec<i32>),
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct SavedPolygonConfig {
    pub id: String,
    pub name: String,
    pub dimensions: (i32, i32), // (width, height) in pixels
}

fn main() -> io::Result<()> {
    let json_str = include_str!("../../backup/motion_path_data_backup_final.json");
    let saved_state: SavedState = from_str(json_str).unwrap();

    let mut train_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("train.csv")?);
    let mut test_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("test.csv")?);

    let num_sequences = saved_state.sequences.len();
    let train_size = (num_sequences as f64 * 0.8) as usize;

    for (i, sequence) in saved_state.sequences.iter().enumerate() {
        let mut polygon_index = 0;

        let mut poly_x = 0.0;
        let mut poly_y = 0.0;
        sequence.active_polygons.iter().for_each(|p| {
            for animation_data in &sequence.polygon_motion_paths {
                if animation_data.polygon_id == p.id {
                    for animation_property in &animation_data.properties {
                        if animation_property.name == "Position" {
                            for (j, keyframe) in animation_property.keyframes.iter().enumerate() {
                                if keyframe.time == Duration::from_secs(5) {
                                    let (x, y) = match &keyframe.value {
                                        KeyframeValue::Position(position) => {
                                            (position[0] as f64, position[1] as f64)
                                        }
                                        _ => (0.0, 0.0),
                                    };
                                    poly_x = x;
                                    poly_y = y;
                                }
                            }
                        }
                    }
                }
            }

            let record = vec![
                format!("{}", polygon_index),
                format!("{}", 5),
                format!("{}", p.dimensions.0),
                format!("{}", p.dimensions.1),
                format!("{}", poly_x),
                format!("{}", poly_y),
            ];

            if i < train_size {
                train_writer
                    .write_record(&record)
                    .expect("Coudn't write header record");
            } else {
                test_writer
                    .write_record(&record)
                    .expect("Coudn't write header record");
            }
            polygon_index = polygon_index + 1;
        });

        for animation_data in &sequence.polygon_motion_paths {
            for animation_property in &animation_data.properties {
                if animation_property.name == "Position" {
                    for (j, keyframe) in animation_property.keyframes.iter().enumerate() {
                        // let time = keyframe.time.secs as f64 + keyframe.time.nanos as f64 / 1e9;
                        let time = keyframe.time;
                        let (x, y) = match &keyframe.value {
                            KeyframeValue::Position(position) => {
                                (position[0] as f64, position[1] as f64)
                            }
                            _ => (0.0, 0.0),
                        };
                        let polygon_id = animation_data.polygon_id.clone();

                        let polygon_data = sequence
                            .active_polygons
                            .iter()
                            .find(|p| p.id == polygon_id)
                            .expect("Couldn't find matching polygon");
                        let polygon_index = sequence
                            .active_polygons
                            .iter()
                            .position(|p| p.id == polygon_id)
                            .expect("Couldn't find matching polygon");

                        let width = polygon_data.dimensions.0;
                        let height = polygon_data.dimensions.1;

                        let record = vec![
                            format!("{}", polygon_index),
                            format!("{}", time.as_secs_f32()),
                            format!("{}", width),
                            format!("{}", height),
                            format!("{}", x),
                            format!("{}", y),
                        ];

                        if i < train_size {
                            train_writer.write_record(&record)?;
                        } else {
                            test_writer.write_record(&record)?;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
