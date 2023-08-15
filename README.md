# Preprocessing the information

1. The very first step is to recollect the data from all the montlhy surveys.
2. Modify the column's names (because we have two rows as header that make gropus, because each column that is a question in the survey has a type depending on the possible answers. We have *Response*, *Otro (especifique)*, *Open-Ended Response*).

	For example: 
	* Especialidad: (Response)
	* Especialidad: Otro (especifique)

	For all those that are *Open-Ended Responses* we don't modify
	anything.

3. After we modify the column's names until it is possible, we separate data, that is already structured from the data that it is not structured yet. This will allow us to analyze data in separate ways, because the type of analysis differs from structured data (descriptive) to the non-structured data that requires som NLP to understand the feelings and emotions or the nature of the response.

