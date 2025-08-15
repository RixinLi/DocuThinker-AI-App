/**
 * GraphQL schema definition
 * @type {string} GraphQL schema definition
 */
const typeDefs = `
  type User {
    id: ID!
    email: String!
    createdAt: String
    documents: [Document!]
  }

  type Document {
    id: ID!
    title: [String!]!
    summary: String!
    originalText: String!
  }

  type CustomUser{
    id: ID!
    email: String
  }

  type Query {
    getUser(id: ID!): User
    getCustomUser(id: ID!): CustomUser
    getDocument(userId: ID!, docId: ID!): Document
    listDocuments(userId: ID!): [Document!]
  }

  type Mutation {
    createUser(email: String!, password: String!): User
    deleteDocument(userId: ID!, docId: ID!): Boolean
    updateDocumentTitle(userId: ID!, docId: ID!, title: String!): Document
  }
`;

module.exports = typeDefs;
