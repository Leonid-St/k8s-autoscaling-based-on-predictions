---
editable: false
sourcePath: en/_api-ref/datalens/function-ref/MCOUNT.md
---

# MCOUNT (window)



#### Syntax {#syntax}

{% list tabs %}

- Standard

  ```
  MCOUNT( value, rows_1 [ , rows_2 ] )
  ```

- Extended

  ```
  MCOUNT( value, rows_1 [ , rows_2 ]
          [ TOTAL | WITHIN ... | AMONG ... ]
          [ ORDER BY ... ]
          [ BEFORE FILTER BY ... ]
        )
  ```

  More info:
  - [TOTAL, WITHIN, AMONG](window-functions.md#syntax-grouping)
  - [ORDER BY](window-functions.md#syntax-order-by)
  - [BEFORE FILTER BY](window-functions.md#syntax-before-filter-by)

{% endlist %}

#### Description {#description}

{% note warning %}

The sorting order is based on the fields listed in the sorting section of the chart and in the `ORDER BY` clause. First, `ORDER BY` fields are used, and then they are complemented by the fields from the chart.

{% endnote %}

Returns the moving count of (non-`NULL`) values in a fixed-size window defined by the sort order and arguments:

| `rows_1`   | `rows_2`   | Window                                                                |
|:-----------|:-----------|:----------------------------------------------------------------------|
| positive   | -          | The current row and `rows_1` preceding rows.                          |
| negative   | -          | The current row and -`rows_1` following rows.                         |
| any sign   | any sign   | `rows_1` preceding rows, the current row and `rows_2` following rows. |


Window functions with a similar behavior: [MSUM](MSUM.md), [MMIN](MMIN.md), [MMAX](MMAX.md), [MAVG](MAVG.md).

See also [COUNT](COUNT.md), [RCOUNT](RCOUNT.md).

**Argument types:**
- `value` — `Any`
- `rows_1` — `Integer`
- `rows_2` — `Integer`


**Return type**: Same type as (`value`)

{% note info %}

Only constant values are accepted for the arguments (`rows_1`, `rows_2`).

{% endnote %}


#### Examples {#examples}

{% cut "Example with two and three arguments" %}


Source data

| **Date**       | **City**          | **Category**        | **Orders**   | **Profit**   |
|:---------------|:------------------|:--------------------|:-------------|:-------------|
| `'2019-03-01'` | `'London'`        | `'Office Supplies'` | `8`          | `120.80`     |
| `'2019-03-04'` | `'London'`        | `'Office Supplies'` | `2`          | `100.00`     |
| `'2019-03-05'` | `'London'`        | `'Furniture'`       | `1`          | `750.00`     |
| `'2019-03-02'` | `'Moscow'`        | `'Furniture'`       | `2`          | `1250.50`    |
| `'2019-03-03'` | `'Moscow'`        | `'Office Supplies'` | `4`          | `85.00`      |
| `'2019-03-01'` | `'San Francisco'` | `'Office Supplies'` | `23`         | `723.00`     |
| `'2019-03-01'` | `'San Francisco'` | `'Furniture'`       | `1`          | `1000.00`    |
| `'2019-03-03'` | `'San Francisco'` | `'Furniture'`       | `4`          | `4000.00`    |
| `'2019-03-02'` | `'Detroit'`       | `'Furniture'`       | `5`          | `3700.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Office Supplies'` | `25`         | `1200.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Furniture'`       | `2`          | `3500.00`    |

Grouped by `[City]`.

Sorted by `[City]`.

Formulas:

- **City**: `[City]` ;
- **Order Sum**: `SUM([Orders])` ;
- **MCOUNT 1**: `MCOUNT(SUM([Orders]), 1)` ;
- **MCOUNT 2**: `MCOUNT(SUM([Orders]), -2)` ;
- **MCOUNT 3**: `MCOUNT(SUM([Orders]) 1, 1)` .


Result

| **City**          | **Order Sum**   | **MCOUNT 1**   | **MCOUNT 2**   | **MCOUNT 3**   |
|:------------------|:----------------|:---------------|:---------------|:---------------|
| `'Detroit'`       | `32`            | `1`            | `3`            | `2`            |
| `'London'`        | `11`            | `2`            | `3`            | `3`            |
| `'Moscow'`        | `6`             | `2`            | `2`            | `3`            |
| `'San Francisco'` | `28`            | `2`            | `1`            | `2`            |

{% endcut %}

{% cut "Example with ORDER BY" %}


Source data

| **Date**       | **City**          | **Category**        | **Orders**   | **Profit**   |
|:---------------|:------------------|:--------------------|:-------------|:-------------|
| `'2019-03-01'` | `'London'`        | `'Office Supplies'` | `8`          | `120.80`     |
| `'2019-03-04'` | `'London'`        | `'Office Supplies'` | `2`          | `100.00`     |
| `'2019-03-05'` | `'London'`        | `'Furniture'`       | `1`          | `750.00`     |
| `'2019-03-02'` | `'Moscow'`        | `'Furniture'`       | `2`          | `1250.50`    |
| `'2019-03-03'` | `'Moscow'`        | `'Office Supplies'` | `4`          | `85.00`      |
| `'2019-03-01'` | `'San Francisco'` | `'Office Supplies'` | `23`         | `723.00`     |
| `'2019-03-01'` | `'San Francisco'` | `'Furniture'`       | `1`          | `1000.00`    |
| `'2019-03-03'` | `'San Francisco'` | `'Furniture'`       | `4`          | `4000.00`    |
| `'2019-03-02'` | `'Detroit'`       | `'Furniture'`       | `5`          | `3700.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Office Supplies'` | `25`         | `1200.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Furniture'`       | `2`          | `3500.00`    |

Grouped by `[City]`.

Sorted by `[City]`.

Formulas:

- **City**: `[City]` ;
- **Order Sum**: `SUM([Orders])` ;
- **MCOUNT 1**: `MCOUNT(SUM([Orders]), 1 ORDER BY [City] DESC)` ;
- **MCOUNT 2**: `MCOUNT(SUM([Orders]), 1 ORDER BY [Order Sum])` .


Result

| **City**          | **Order Sum**   | **MCOUNT 1**   | **MCOUNT 2**   |
|:------------------|:----------------|:---------------|:---------------|
| `'Detroit'`       | `32`            | `2`            | `2`            |
| `'London'`        | `11`            | `2`            | `2`            |
| `'Moscow'`        | `6`             | `2`            | `1`            |
| `'San Francisco'` | `28`            | `1`            | `2`            |

{% endcut %}

{% cut "Example with grouping" %}


Source data

| **Date**       | **City**          | **Category**        | **Orders**   | **Profit**   |
|:---------------|:------------------|:--------------------|:-------------|:-------------|
| `'2019-03-01'` | `'London'`        | `'Office Supplies'` | `8`          | `120.80`     |
| `'2019-03-04'` | `'London'`        | `'Office Supplies'` | `2`          | `100.00`     |
| `'2019-03-05'` | `'London'`        | `'Furniture'`       | `1`          | `750.00`     |
| `'2019-03-02'` | `'Moscow'`        | `'Furniture'`       | `2`          | `1250.50`    |
| `'2019-03-03'` | `'Moscow'`        | `'Office Supplies'` | `4`          | `85.00`      |
| `'2019-03-01'` | `'San Francisco'` | `'Office Supplies'` | `23`         | `723.00`     |
| `'2019-03-01'` | `'San Francisco'` | `'Furniture'`       | `1`          | `1000.00`    |
| `'2019-03-03'` | `'San Francisco'` | `'Furniture'`       | `4`          | `4000.00`    |
| `'2019-03-02'` | `'Detroit'`       | `'Furniture'`       | `5`          | `3700.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Office Supplies'` | `25`         | `1200.00`    |
| `'2019-03-04'` | `'Detroit'`       | `'Furniture'`       | `2`          | `3500.00`    |

Grouped by `[City]`, `[Category]`.

Sorted by `[City]`, `[Category]`.

Formulas:

- **City**: `[City]` ;
- **Category**: `[Category]` ;
- **Order Sum**: `SUM([Orders])` ;
- **MCOUNT 1**: `MCOUNT(SUM([Orders]), 1 TOTAL ORDER BY [City], [Category])` ;
- **MCOUNT 2**: `MCOUNT(SUM([Orders]), 1 WITHIN [City] ORDER BY [Category])` ;
- **MCOUNT 3**: `MCOUNT(SUM([Orders]), 1 AMONG [City] ORDER BY [City])` .


Result

| **City**          | **Category**        | **Order Sum**   | **MCOUNT 1**   | **MCOUNT 2**   | **MCOUNT 3**   |
|:------------------|:--------------------|:----------------|:---------------|:---------------|:---------------|
| `'Detroit'`       | `'Furniture'`       | `7`             | `1`            | `1`            | `1`            |
| `'Detroit'`       | `'Office Supplies'` | `25`            | `2`            | `2`            | `1`            |
| `'London'`        | `'Furniture'`       | `1`             | `2`            | `1`            | `2`            |
| `'London'`        | `'Office Supplies'` | `10`            | `2`            | `2`            | `2`            |
| `'Moscow'`        | `'Furniture'`       | `2`             | `2`            | `1`            | `2`            |
| `'Moscow'`        | `'Office Supplies'` | `4`             | `2`            | `2`            | `2`            |
| `'San Francisco'` | `'Furniture'`       | `5`             | `2`            | `1`            | `2`            |
| `'San Francisco'` | `'Office Supplies'` | `23`            | `2`            | `2`            | `2`            |

{% endcut %}


#### Data source support {#data-source-support}

`ClickHouse 21.8`, `Files`, `Google Sheets`, `Microsoft SQL Server 2017 (14.0)`, `MySQL 5.7`, `Oracle Database 12c (12.1)`, `PostgreSQL 9.3`, `Yandex Documents`.
