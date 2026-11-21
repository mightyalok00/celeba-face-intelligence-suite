-- ==============================================================================
-- CELEBA DATABASE SETUP & ARCHITECTURE SCRIPT
-- File: D:\scan\database\celeba_db_setup.sql
-- Database: celeba_db
-- ==============================================================================
-- Complete initialization script:
-- 1. Database Creation & Settings
-- 2. Master & Child Normalized Tables (celebrities, images_metadata, celeba_identities, celeba_attributes)
-- 3. Foreign Key Constraints (Enables Workbench EER Diagram relationship lines)
-- 4. Data Loading Pipelines (CSV and TXT ingestion)
-- 5. Physical Master Table: celeba_combined_dataset (All 202,599 rows with indexes)
-- 6. Unified Relational View: celeba_master_view
-- 7. Verification and Diagnostics
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- PART 1: DATABASE INITIALIZATION
-- ------------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS celeba_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE celeba_db;

-- ------------------------------------------------------------------------------
-- PART 2: CORE NORMALIZED TABLES
-- ------------------------------------------------------------------------------

-- 1. Master Celebrities Reference Table
DROP TABLE IF EXISTS celeba_combined_dataset;
DROP TABLE IF EXISTS celeba_attributes;
DROP TABLE IF EXISTS celeba_identities;
DROP TABLE IF EXISTS images_metadata;
DROP TABLE IF EXISTS celebrities;

CREATE TABLE celebrities (
    celebrity_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for celebrity',
    celebrity_name VARCHAR(150) NOT NULL COMMENT 'Full name of the celebrity',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Record timestamp',
    INDEX idx_celeb_name (celebrity_name)
) ENGINE=InnoDB COMMENT='Master reference table for celebrity names';

-- 2. Central Image Metadata Table
CREATE TABLE images_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Primary surrogate key',
    image_id VARCHAR(32) NOT NULL COMMENT 'Unique file identifier e.g. 000001.jpg',
    file_name VARCHAR(64) NOT NULL COMMENT 'Filename in img_align_celeba e.g. 000001.png',
    file_path VARCHAR(255) NOT NULL COMMENT 'Full disk path on D: drive',
    person_id INT NOT NULL DEFAULT 0 COMMENT 'Ground-truth person cluster ID',
    celebrity_id INT NOT NULL COMMENT 'Foreign key to celebrities table',
    file_format VARCHAR(10) DEFAULT 'png' COMMENT 'Image extension',
    UNIQUE INDEX uq_meta_image_id (image_id),
    INDEX idx_celeb_id (celebrity_id),
    INDEX idx_person_id (person_id)
) ENGINE=InnoDB COMMENT='Metadata and disk location for every face image';

-- 3. Ground-Truth Person Identities Table (10,177 clusters)
CREATE TABLE celeba_identities (
    image_id VARCHAR(32) NOT NULL PRIMARY KEY COMMENT 'Matches image_id in images_metadata',
    person_id INT NOT NULL COMMENT 'Identity cluster identifier (1 - 10,177)',
    INDEX idx_pid (person_id)
) ENGINE=InnoDB COMMENT='Official CelebA ground-truth identity annotations';

-- 4. 40 Binary Facial Attribute Annotations Table
CREATE TABLE celeba_attributes (
    image_id VARCHAR(32) NOT NULL PRIMARY KEY COMMENT 'Matches image_id in images_metadata',
    `5_o_Clock_Shadow` TINYINT NOT NULL DEFAULT -1,
    `Arched_Eyebrows` TINYINT NOT NULL DEFAULT -1,
    `Attractive` TINYINT NOT NULL DEFAULT -1,
    `Bags_Under_Eyes` TINYINT NOT NULL DEFAULT -1,
    `Bald` TINYINT NOT NULL DEFAULT -1,
    `Bangs` TINYINT NOT NULL DEFAULT -1,
    `Big_Lips` TINYINT NOT NULL DEFAULT -1,
    `Big_Nose` TINYINT NOT NULL DEFAULT -1,
    `Black_Hair` TINYINT NOT NULL DEFAULT -1,
    `Blond_Hair` TINYINT NOT NULL DEFAULT -1,
    `Blurry` TINYINT NOT NULL DEFAULT -1,
    `Brown_Hair` TINYINT NOT NULL DEFAULT -1,
    `Bushy_Eyebrows` TINYINT NOT NULL DEFAULT -1,
    `Chubby` TINYINT NOT NULL DEFAULT -1,
    `Double_Chin` TINYINT NOT NULL DEFAULT -1,
    `Eyeglasses` TINYINT NOT NULL DEFAULT -1,
    `Goatee` TINYINT NOT NULL DEFAULT -1,
    `Gray_Hair` TINYINT NOT NULL DEFAULT -1,
    `Heavy_Makeup` TINYINT NOT NULL DEFAULT -1,
    `High_Cheekbones` TINYINT NOT NULL DEFAULT -1,
    `Male` TINYINT NOT NULL DEFAULT -1,
    `Mouth_Slightly_Open` TINYINT NOT NULL DEFAULT -1,
    `Mustache` TINYINT NOT NULL DEFAULT -1,
    `Narrow_Eyes` TINYINT NOT NULL DEFAULT -1,
    `No_Beard` TINYINT NOT NULL DEFAULT -1,
    `Oval_Face` TINYINT NOT NULL DEFAULT -1,
    `Pale_Skin` TINYINT NOT NULL DEFAULT -1,
    `Pointy_Nose` TINYINT NOT NULL DEFAULT -1,
    `Receding_Hairline` TINYINT NOT NULL DEFAULT -1,
    `Rosy_Cheeks` TINYINT NOT NULL DEFAULT -1,
    `Sideburns` TINYINT NOT NULL DEFAULT -1,
    `Smiling` TINYINT NOT NULL DEFAULT -1,
    `Straight_Hair` TINYINT NOT NULL DEFAULT -1,
    `Wavy_Hair` TINYINT NOT NULL DEFAULT -1,
    `Wearing_Earrings` TINYINT NOT NULL DEFAULT -1,
    `Wearing_Hat` TINYINT NOT NULL DEFAULT -1,
    `Wearing_Lipstick` TINYINT NOT NULL DEFAULT -1,
    `Wearing_Necklace` TINYINT NOT NULL DEFAULT -1,
    `Wearing_Necktie` TINYINT NOT NULL DEFAULT -1,
    `Young` TINYINT NOT NULL DEFAULT -1
) ENGINE=InnoDB COMMENT='CelebA 40 binary attributes (-1: False, 1: True)';

-- ------------------------------------------------------------------------------
-- PART 3: RELATIONAL INTEGRITY & FOREIGN KEYS (WORKBENCH EER DIAGRAM)
-- ------------------------------------------------------------------------------

-- FK 1: images_metadata -> celebrities (1 : N)
ALTER TABLE images_metadata 
    ADD CONSTRAINT fk_metadata_celebrities 
    FOREIGN KEY (celebrity_id) REFERENCES celebrities(celebrity_id)
    ON DELETE RESTRICT 
    ON UPDATE CASCADE;

-- FK 2: celeba_identities -> images_metadata (1 : 1)
ALTER TABLE celeba_identities 
    ADD CONSTRAINT fk_identities_metadata 
    FOREIGN KEY (image_id) REFERENCES images_metadata(image_id)
    ON DELETE CASCADE 
    ON UPDATE CASCADE;

-- FK 3: celeba_attributes -> images_metadata (1 : 1)
ALTER TABLE celeba_attributes 
    ADD CONSTRAINT fk_attributes_metadata 
    FOREIGN KEY (image_id) REFERENCES images_metadata(image_id)
    ON DELETE CASCADE 
    ON UPDATE CASCADE;

-- ------------------------------------------------------------------------------
-- PART 4: DATA LOADING PIPELINES (PORTABLE / CROSS-PLATFORM)
-- ------------------------------------------------------------------------------
-- Note: Paths use portable relative resolution from project root.
-- For custom environments, you can define session variables:
--   SET @celeba_data_dir = '/path/to/project/data/metadata';
--   SET @celeba_img_dir  = '/path/to/project/img_align_celeba';

-- 1. Load Celebrities Reference Data
-- Note: If running via CLI, ensure --local-infile=1 is enabled.
LOAD DATA LOCAL INFILE 'data/metadata/celebrities.csv'
INTO TABLE celebrities
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(celebrity_id, celebrity_name);

-- 2. Load CelebA Attributes (202,599 records)
LOAD DATA LOCAL INFILE 'data/metadata/list_attr_celeba.csv'
INTO TABLE celeba_attributes
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

-- 3. Load CelebA Ground-Truth Person Identities (202,599 records)
LOAD DATA LOCAL INFILE 'data/metadata/identity_CelebA.txt'
INTO TABLE celeba_identities
FIELDS TERMINATED BY ' '
LINES TERMINATED BY '\n'
(image_id, person_id);

-- 4. Populate ground-truth identity records for all 10,177 CelebA identities
-- Note: identity_CelebA.txt contains anonymized integer IDs (1 to 10,177) created by CUHK.
INSERT INTO celebrities (celebrity_id, celebrity_name)
SELECT DISTINCT 
    person_id AS celebrity_id, 
    CONCAT('CelebA Identity #', person_id) AS celebrity_name 
FROM celeba_identities
ON DUPLICATE KEY UPDATE celebrity_name = CONCAT('CelebA Identity #', VALUES(celebrity_id));

-- 5. Populate images_metadata linking to identities and celebrities (portable path)
INSERT INTO images_metadata (image_id, file_name, file_path, person_id, celebrity_id, file_format)
SELECT 
    a.image_id,
    REPLACE(a.image_id, '.jpg', '.png') AS file_name,
    CONCAT('img_align_celeba/', REPLACE(a.image_id, '.jpg', '.png')) AS file_path,
    COALESCE(i.person_id, 0) AS person_id,
    COALESCE(i.person_id, 0) AS celebrity_id,
    'png' AS file_format
FROM celeba_attributes a
LEFT JOIN celeba_identities i ON a.image_id = i.image_id;

-- ------------------------------------------------------------------------------
-- PART 5: MATERIALIZED MASTER TABLE (celeba_combined_dataset)
-- ------------------------------------------------------------------------------
-- Combines all 4 tables into one physical, highly-indexed analytical master table
DROP TABLE IF EXISTS celeba_combined_dataset;

CREATE TABLE celeba_combined_dataset AS
SELECT 
    -- Primary & File Identifiers
    m.id AS metadata_id,
    m.image_id,
    m.file_name,
    m.file_path,
    m.file_format,

    -- Identity & Celebrity
    COALESCE(i.person_id, m.person_id) AS person_id,
    COALESCE(c.celebrity_id, m.person_id) AS celebrity_id,
    COALESCE(c.celebrity_name, CONCAT('CelebA Identity #', COALESCE(i.person_id, m.person_id))) AS celebrity_name,

    -- Formatted Demographic Labels
    CASE WHEN a.Male = 1 THEN 'Male' ELSE 'Female' END AS gender,
    CASE WHEN a.Young = 1 THEN 'Young' ELSE 'Senior' END AS age_group,
    CASE WHEN a.Smiling = 1 THEN 'Smiling' ELSE 'Neutral' END AS expression,
    CASE WHEN a.Eyeglasses = 1 THEN 'Yes' ELSE 'No' END AS wearing_glasses,

    -- All 40 Raw CelebA Attributes (-1: False, 1: True)
    a.`5_o_Clock_Shadow`,
    a.`Arched_Eyebrows`,
    a.`Attractive`,
    a.`Bags_Under_Eyes`,
    a.`Bald`,
    a.`Bangs`,
    a.`Big_Lips`,
    a.`Big_Nose`,
    a.`Black_Hair`,
    a.`Blond_Hair`,
    a.`Blurry`,
    a.`Brown_Hair`,
    a.`Bushy_Eyebrows`,
    a.`Chubby`,
    a.`Double_Chin`,
    a.`Eyeglasses`,
    a.`Goatee`,
    a.`Gray_Hair`,
    a.`Heavy_Makeup`,
    a.`High_Cheekbones`,
    a.`Male`,
    a.`Mouth_Slightly_Open`,
    a.`Mustache`,
    a.`Narrow_Eyes`,
    a.`No_Beard`,
    a.`Oval_Face`,
    a.`Pale_Skin`,
    a.`Pointy_Nose`,
    a.`Receding_Hairline`,
    a.`Rosy_Cheeks`,
    a.`Sideburns`,
    a.`Smiling`,
    a.`Straight_Hair`,
    a.`Wavy_Hair`,
    a.`Wearing_Earrings`,
    a.`Wearing_Hat`,
    a.`Wearing_Lipstick`,
    a.`Wearing_Necklace`,
    a.`Wearing_Necktie`,
    a.`Young`
FROM images_metadata m
INNER JOIN celeba_attributes a ON m.image_id = a.image_id
LEFT JOIN celeba_identities i ON m.image_id = i.image_id
LEFT JOIN celebrities c ON m.celebrity_id = c.celebrity_id;

-- Add Primary Key and High-Performance Query Indexes
ALTER TABLE celeba_combined_dataset
    ADD PRIMARY KEY (metadata_id),
    ADD UNIQUE INDEX uq_comb_image_id (image_id),
    ADD INDEX idx_comb_person (person_id),
    ADD INDEX idx_comb_celeb_name (celebrity_name),
    ADD INDEX idx_comb_gender (gender),
    ADD INDEX idx_comb_smiling (Smiling),
    ADD INDEX idx_comb_male (Male),
    ADD INDEX idx_comb_young (Young);

-- Connect combined master table to images_metadata for EER Diagram
ALTER TABLE celeba_combined_dataset
    ADD CONSTRAINT fk_combined_metadata
    FOREIGN KEY (image_id) REFERENCES images_metadata(image_id)
    ON DELETE CASCADE
    ON UPDATE CASCADE;

-- ------------------------------------------------------------------------------
-- PART 6: UNIFIED RELATIONAL VIEW (celeba_master_view)
-- ------------------------------------------------------------------------------
DROP VIEW IF EXISTS celeba_master_view;

CREATE VIEW celeba_master_view AS
SELECT 
    m.id AS metadata_id,
    m.image_id,
    m.file_name,
    m.file_path,
    m.file_format,
    m.person_id,
    c.celebrity_id,
    c.celebrity_name,
    CASE WHEN a.Male = 1 THEN 'Male' ELSE 'Female' END AS gender,
    CASE WHEN a.Young = 1 THEN 'Young' ELSE 'Senior' END AS age_group,
    CASE WHEN a.Smiling = 1 THEN 'Smiling' ELSE 'Neutral/Serious' END AS facial_expression,
    CASE WHEN a.Eyeglasses = 1 THEN 'Yes' ELSE 'No' END AS wearing_glasses,
    CASE 
        WHEN a.Blond_Hair = 1 THEN 'Blond'
        WHEN a.Black_Hair = 1 THEN 'Black'
        WHEN a.Brown_Hair = 1 THEN 'Brown'
        WHEN a.Gray_Hair = 1 THEN 'Gray'
        WHEN a.Bald = 1 THEN 'Bald'
        ELSE 'Other/Covered'
    END AS primary_hair_color,
    a.Attractive,
    a.Bangs,
    a.Bushy_Eyebrows,
    a.Heavy_Makeup,
    a.High_Cheekbones,
    a.No_Beard,
    a.Smiling,
    a.Male,
    a.Young,
    a.Eyeglasses
FROM images_metadata m
INNER JOIN celebrities c ON m.celebrity_id = c.celebrity_id
INNER JOIN celeba_attributes a ON m.image_id = a.image_id;

-- ------------------------------------------------------------------------------
-- PART 7: VERIFICATION & AUDIT
-- ------------------------------------------------------------------------------
SELECT 
    'celebrities' AS table_name, COUNT(*) AS total_rows FROM celebrities
UNION ALL
SELECT 
    'images_metadata', COUNT(*) FROM images_metadata
UNION ALL
SELECT 
    'celeba_identities', COUNT(*) FROM celeba_identities
UNION ALL
SELECT 
    'celeba_attributes', COUNT(*) FROM celeba_attributes
UNION ALL
SELECT 
    'celeba_combined_dataset', COUNT(*) FROM celeba_combined_dataset;

-- Verify Foreign Key Registry for Workbench EER Diagram
SELECT 
    CONSTRAINT_NAME, 
    TABLE_NAME, 
    COLUMN_NAME, 
    REFERENCED_TABLE_NAME, 
    REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'celeba_db' 
  AND REFERENCED_TABLE_NAME IS NOT NULL;

-- ------------------------------------------------------------------------------
-- PART 8: LEAST-PRIVILEGE SECURITY CONFIGURATION
-- ------------------------------------------------------------------------------
-- Dedicated application service user with restricted permissions
-- (Replaces direct root usage for production security compliance)
CREATE USER IF NOT EXISTS 'celeba_user'@'%' IDENTIFIED BY 'celeba_pass_2026';
CREATE USER IF NOT EXISTS 'celeba_user'@'localhost' IDENTIFIED BY 'celeba_pass_2026';
GRANT SELECT, INSERT, UPDATE, EXECUTE ON celeba_db.* TO 'celeba_user'@'%';
GRANT SELECT, INSERT, UPDATE, EXECUTE ON celeba_db.* TO 'celeba_user'@'localhost';
FLUSH PRIVILEGES;

