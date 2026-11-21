-- ==============================================================================
-- CELEBA SQL QUERIES, JOINS, DUPLICATE OPERATIONS & ANALYTICS
-- File: D:\scan\database\celeba_queries_and_analytics.sql
-- Database: celeba_db
-- ==============================================================================
-- Streamlined reference and lab exercises:
-- 1. SQL JOIN Mastery: INNER, LEFT, RIGHT, FULL OUTER (emulated), SELF, CROSS
-- 2. Duplicate Detection: GROUP BY + HAVING, and Window Functions (ROW_NUMBER)
-- 3. Duplicate Deletion Strategies: Multi-Table Self-Join, Subquery, and CTE methods
-- 4. Analytical & Business Intelligence Queries on celeba_combined_dataset
-- ==============================================================================

USE celeba_db;

-- ==============================================================================
-- PART 1: SQL JOIN OPERATIONS
-- ==============================================================================

-- 1.1 INNER JOIN: Match Celebrities, Images, and Key Facial Attributes
SELECT 
    c.celebrity_id,
    UPPER(c.celebrity_name) AS celebrity_name,
    m.image_id,
    m.file_name,
    m.file_path,
    CASE WHEN a.Smiling = 1 THEN 'Smiling' ELSE 'Neutral' END AS expression,
    CASE WHEN a.Male = 1 THEN 'Male' ELSE 'Female' END AS gender,
    CASE WHEN a.Young = 1 THEN 'Young' ELSE 'Senior' END AS age_group,
    CASE WHEN a.Eyeglasses = 1 THEN 'Yes' ELSE 'No' END AS wearing_glasses
FROM celebrities c
INNER JOIN images_metadata m ON c.celebrity_id = m.celebrity_id
INNER JOIN celeba_attributes a ON m.image_id = a.image_id
WHERE c.celebrity_name IN ('CelebA Identity #1', 'CelebA Identity #2', 'CelebA Identity #8', 'CelebA Identity #973')
ORDER BY c.celebrity_name, m.file_name
LIMIT 20;

-- 1.2 LEFT JOIN: Include ALL Celebrities and Count Associated Photos
SELECT 
    c.celebrity_id,
    c.celebrity_name,
    COUNT(m.id) AS total_photos,
    COALESCE(GROUP_CONCAT(m.file_name ORDER BY m.file_name SEPARATOR ', '), 'None') AS sample_photos
FROM celebrities c
LEFT JOIN images_metadata m ON c.celebrity_id = m.celebrity_id
GROUP BY c.celebrity_id, c.celebrity_name
ORDER BY total_photos DESC, c.celebrity_name ASC
LIMIT 20;

-- 1.3 RIGHT JOIN: Verify Attribute Records Have Matching Metadata
SELECT 
    m.file_name,
    m.file_path,
    c.celebrity_name,
    a.Attractive,
    a.Heavy_Makeup,
    a.Smiling
FROM celebrities c
INNER JOIN images_metadata m ON c.celebrity_id = m.celebrity_id
RIGHT JOIN celeba_attributes a ON m.image_id = a.image_id
LIMIT 10;

-- 1.4 FULL OUTER JOIN (Canonical UNION ALL + Anti-Join Emulation)
-- High-performance emulation: avoids in-memory temporary table deduplication
SELECT c.celebrity_id, c.celebrity_name, m.image_id, m.file_name
FROM celebrities c
LEFT JOIN images_metadata m ON c.celebrity_id = m.celebrity_id
WHERE c.celebrity_id <= 10

UNION ALL

SELECT c.celebrity_id, c.celebrity_name, m.image_id, m.file_name
FROM images_metadata m
LEFT JOIN celebrities c ON m.celebrity_id = c.celebrity_id
WHERE m.id <= 10 AND c.celebrity_id IS NULL;

-- 1.5 SELF JOIN: Pairwise Comparison of Photos for the Same Celebrity
-- img1.id < img2.id eliminates self-comparison and symmetric duplicates
SELECT 
    c.celebrity_name,
    img1.file_name AS photo_1,
    img2.file_name AS photo_2,
    CONCAT('Comparing photos for ', c.celebrity_name) AS note
FROM images_metadata img1
INNER JOIN images_metadata img2 
    ON img1.celebrity_id = img2.celebrity_id AND img1.id < img2.id
INNER JOIN celebrities c ON img1.celebrity_id = c.celebrity_id
WHERE c.celebrity_name = 'CelebA Identity #8'
LIMIT 10;

-- 1.6 CROSS JOIN: Cartesian Product of Sample Celebrities and Target Attributes
SELECT 
    c.celebrity_name,
    attr.label AS target_attribute
FROM (SELECT celebrity_name FROM celebrities LIMIT 3) AS c
CROSS JOIN (
    SELECT 'Smiling' AS label
    UNION ALL SELECT 'Eyeglasses'
    UNION ALL SELECT 'Wearing_Hat'
    UNION ALL SELECT 'Bangs'
) AS attr;


-- ==============================================================================
-- PART 2: DUPLICATE DETECTION METHODS
-- ==============================================================================

-- 2.1 Detect Duplicate Images by file_name using GROUP BY & HAVING
SELECT 
    file_name,
    COUNT(*) AS occurrences,
    MIN(id) AS lowest_id,
    MAX(id) AS highest_id
FROM images_metadata
GROUP BY file_name
HAVING COUNT(*) > 1;

-- 2.2 Number Duplicate Identity Appearances using Window Function ROW_NUMBER()
SELECT 
    image_id,
    person_id,
    ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY image_id ASC) AS photo_instance_index
FROM celeba_identities
LIMIT 25;

-- 2.3 Flag Duplicate Occurrences without Collapsing Rows using COUNT(*) OVER()
SELECT 
    id,
    image_id,
    celebrity_id,
    COUNT(*) OVER (PARTITION BY celebrity_id) AS total_photos_for_celebrity
FROM images_metadata
LIMIT 25;


-- ==============================================================================
-- PART 3: DUPLICATE DELETION STRATEGIES (DEMO ENVIRONMENT)
-- ==============================================================================

DROP TABLE IF EXISTS demo_image_records;
CREATE TABLE demo_image_records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id VARCHAR(32) NOT NULL,
    celebrity_id INT NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO demo_image_records (image_id, celebrity_id, file_path)
VALUES 
    ('000001.jpg', 1, 'D:/scan/img_align_celeba/000001.png'),
    ('000001.jpg', 1, 'D:/scan/img_align_celeba/000001.png'),
    ('000001.jpg', 1, 'D:/scan/img_align_celeba/000001.png'),
    ('000002.jpg', 2, 'D:/scan/img_align_celeba/000002.png'),
    ('000002.jpg', 2, 'D:/scan/img_align_celeba/000002.png'),
    ('000003.jpg', 3, 'D:/scan/img_align_celeba/000003.png');

-- METHOD A: DELETE USING SELF-JOIN (Multi-Table Delete)
DELETE t1
FROM demo_image_records t1
INNER JOIN demo_image_records t2 
    ON t1.image_id = t2.image_id AND t1.record_id > t2.record_id;

SELECT 'After Method A (Self-Join Delete)' AS method, record_id, image_id FROM demo_image_records;

-- Reset demo duplicates for Method B
INSERT INTO demo_image_records (image_id, celebrity_id, file_path)
VALUES 
    ('000001.jpg', 1, 'D:/scan/img_align_celeba/000001.png'),
    ('000002.jpg', 2, 'D:/scan/img_align_celeba/000002.png');

-- METHOD B: DELETE USING SUBQUERY WITH MIN()
DELETE FROM demo_image_records
WHERE record_id NOT IN (
    SELECT min_id FROM (
        SELECT MIN(record_id) AS min_id FROM demo_image_records GROUP BY image_id
    ) AS keep
);

SELECT 'After Method B (Subquery MIN Delete)' AS method, record_id, image_id FROM demo_image_records;

-- Reset demo duplicates for Method C
INSERT INTO demo_image_records (image_id, celebrity_id, file_path)
VALUES ('000001.jpg', 1, 'D:/scan/img_align_celeba/000001.png');

-- METHOD C: DELETE USING CTE & ROW_NUMBER()
WITH DuplicatesCTE AS (
    SELECT 
        record_id,
        ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY record_id ASC) AS rn
    FROM demo_image_records
)
DELETE FROM demo_image_records
WHERE record_id IN (SELECT record_id FROM DuplicatesCTE WHERE rn > 1);

SELECT 'After Method C (CTE Window Delete)' AS method, record_id, image_id FROM demo_image_records;


-- ==============================================================================
-- PART 4: ANALYTICS & BUSINESS INTELLIGENCE (ON celeba_combined_dataset)
-- ==============================================================================

-- 4.1 Top Celebrities with Photo Counts & Smile Percentages
SELECT 
    celebrity_id,
    celebrity_name,
    COUNT(*) AS total_photos,
    SUM(CASE WHEN expression = 'Smiling' THEN 1 ELSE 0 END) AS smiling_count,
    SUM(CASE WHEN wearing_glasses = 'Yes' THEN 1 ELSE 0 END) AS glasses_count,
    ROUND(AVG(CASE WHEN expression = 'Smiling' THEN 100.0 ELSE 0.0 END), 1) AS smiling_pct
FROM celeba_combined_dataset
GROUP BY celebrity_id, celebrity_name
ORDER BY total_photos DESC
LIMIT 15;

-- 4.2 Cross-Tabulation: Smiling Rate by Gender and Age Group
SELECT 
    gender,
    age_group,
    COUNT(*) AS sample_count,
    SUM(CASE WHEN expression = 'Smiling' THEN 1 ELSE 0 END) AS smiling_count,
    ROUND(AVG(CASE WHEN expression = 'Smiling' THEN 100.0 ELSE 0.0 END), 2) AS smiling_pct
FROM celeba_combined_dataset
GROUP BY gender, age_group
ORDER BY gender, age_group;

-- 4.3 Co-Occurrence: Glasses and Facial Hair in Males (Optimized with Window Function)
SELECT 
    wearing_glasses,
    CASE WHEN Goatee = 1 OR Mustache = 1 OR Sideburns = 1 THEN 'Has Facial Hair' ELSE 'Clean Shaven' END AS facial_hair,
    COUNT(*) AS total_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage_share
FROM celeba_combined_dataset
WHERE gender = 'Male'
GROUP BY wearing_glasses, facial_hair
ORDER BY total_count DESC;

-- 4.4 High-Performance Filtered Lookup Using Composite Indexes
SELECT 
    metadata_id,
    image_id,
    file_name,
    celebrity_name,
    gender,
    age_group,
    expression,
    wearing_glasses
FROM celeba_combined_dataset
WHERE gender = 'Female' 
  AND age_group = 'Young' 
  AND expression = 'Smiling' 
  AND wearing_glasses = 'No'
ORDER BY metadata_id ASC
LIMIT 20;
